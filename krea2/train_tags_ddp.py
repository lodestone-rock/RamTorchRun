"""train_tags_ddp.py — train rank-32 LoRA + the bit-code tag embedder, DDP.

The base is FROZEN (bf16); the trainable set is the injected LoRA (whole
backbone, rank 32) plus the full-rank TagCodeEmbedder. Data parallel via
ramtorch MultiGPUWrapper (single process, NCCL, ZeRO-1): every GPU holds a
full replica and the frozen components; grads are tiny so all-reduce is
cheap at 1024+ widths.

Conditioning per your spec:
- caption bands: v3's five bands, uniform weight, tags band ALSO plain text
  (treated like any other caption; shuffled as usual, never dropped).
- tag ids: from parquet_dataloader tag_column="tags", fed to TagCodeEmbedder.
- per-tag dropout: each tag token independently dropped w.p. 0.1
  (structured augmentation: zeroing a tag token zeroes its code -> the
  anchor; see TagCodeEmbedder.forward) — implemented by masking the ids to
  the anchor id 0 per-token.
- uncond (uncond_ratio): caption blanked AND the whole tag block dropped
  (bit-exact no-op), matching CFG negative semantics.

Run:
    .venv/bin/python krea2/train_tags_ddp.py krea2/configs/train_tags_ddp.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from einops import rearrange
from safetensors.torch import load_file, save_file
from PIL import Image
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from ramtorch.multi_gpu import MultiGPUWrapper

from krea2.lion import Lion, validate_lion_config
from krea2.model.autoencoder import QwenAutoencoder
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.encoder import Qwen3VLConditioner
from krea2.model.lora import inject_lora
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.sampling import prepare
from krea2.model.tag_code import TagCodeEmbedder
from krea2.train_utils import (
    _mu_from_seq_len, _pin_sdpa_backends, sample_timesteps, vae_decode, vae_encode,
)
from dataloaders.parquet_dataloader import ParquetTextImageDataset

torch.manual_seed(0)


def build_frozen(cfg: dict, devices: list[str]):
    """Teacher DiT (frozen bf16 base), conditioner, VAE — one copy per GPU."""
    dit_cfg = MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")]
    enc_cfg = ENCODER_CONFIGS[cfg.get("encoder_config", "qwen3_vl_4b")]

    print("Loading base...")
    base_sd = load_file(cfg["full_checkpoint"], device="cpu")
    bases = {}
    for dev in devices:
        with torch.device("meta"):
            d = SingleStreamDiT(dit_cfg)
        d.load_state_dict(base_sd, strict=True, assign=True)
        bases[dev] = d.to(dev, dtype=torch.bfloat16).eval().requires_grad_(False)
    del base_sd

    print("Loading conditioner + VAE...")
    encoder_id = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    conditioners, aes = {}, {}
    for dev in devices:
        c = Qwen3VLConditioner(encoder_id, max_length=enc_cfg.max_length)
        conditioners[dev] = c.to(dev, dtype=torch.bfloat16).eval().requires_grad_(False)
        ae = QwenAutoencoder()
        ae.ae = ae.ae.to(torch.bfloat16)
        aes[dev] = ae.to(dev).eval().requires_grad_(False)
    return bases, conditioners, aes, dit_cfg, enc_cfg


def make_train_factory(cfg: dict, dit_cfg, vocab_size: int, code_path: str,
                       vocab_fingerprint: str):
    """Model factory per GPU: frozen-base copy is attached as buffers? No —

    the frozen base must NOT be part of the wrapped module (the wrapper moves
    and casts the module; the base would be needlessly replicated as
    trainable-shaped tensors and get broadcast every step). Instead the
    forward_backward_fn uses wrapper.models[g] for the TRAINABLE student and
    a separately-built frozen base dict for the teacher side... but the
    trainable LoRA must sit ON TOP of the frozen base weights.

    Resolution: the wrapped module IS a full DiT whose Linears carry LoRA;
    non-LoRA params are the frozen base (requires_grad=False after setup via
    requires_grad_(False) on non-LoRA params — done in freeze_non_lora below,
    executed per replica inside fb_fn's first call... simpler: the factory
    freezes non-LoRA params itself, since inject_lora runs in-factory).
    """
    def factory() -> SingleStreamDiT:
        with torch.device("meta"):
            d = SingleStreamDiT(dit_cfg)
        d.load_state_dict(load_file(cfg["full_checkpoint"], device="cpu"),
                          strict=True, assign=True)
        # The wrapper casts every replica to ITS dtype arg (bf16); trainable
        # masters are restored to fp32 after setup() in train(). Do NOT cast
        # here — an fp32 base on the GPU is what OOM'd (51GB x 4).
        inject_lora(d, rank=cfg.get("lora_rank", 32),
                    alpha=float(cfg.get("lora_alpha", cfg.get("lora_rank", 32))),
                    exclude_prefixes=tuple(cfg.get("lora_exclude_prefixes", ())))
        # freeze every non-LoRA param, then attach the tag-code encoder
        for n, p in d.named_parameters():
            p.requires_grad_(False)
        from krea2.model.lora import LoRALinear
        for m in d.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad_(True)
                m.lora_B.requires_grad_(True)
        tc = TagCodeEmbedder(vocab_size, d.config.features,
                             n_bits=int(cfg.get("tag_code", {}).get("n_bits", 30)))
        tc.load_codes(code_path, vocab_fingerprint)
        d.tagcode = tc
        # Activation checkpointing on every block (the repo norm — measured
        # 60GB of saved activations without it at 256px/batch8 on a 12.8B
        # monolith; mmdit.forward honors self.grad_ckpt, txtfusion too).
        d.grad_ckpt = True
        return d
    return factory


@torch.no_grad()
def preview_dataset(model, conditioner, ae, pieces, patch, resolution, steps,
                    out_path, dev, cfg_scale=4.5, mu_y1=0.5, mu_y2=1.15,
                    mu_sigma=1.0, minres=256, maxres=1280, mu_override=None,
                    preview_res=512, seed=0):
    """Dataset-driven preview: rows = conditioning variants, cols = samples.

    ``pieces``: list of per-GPU dicts {images [b,3,H,W] (CPU float [-1,1]),
    captions [b], tag_ids [b,T], tag_mask [b,T]} — the batch each GPU just
    trained on (real data -> the GT row is meaningful). Each GPU renders its
    own b samples in 4 conditioning variants; the caller stitches
    [4, total_b, 3, H, W] across GPUs into the grid.

    Rows (top->bottom):
      1. both — text caption + real tag embeddings, CFG
      2. text — caption only, tag block all-anchor, CFG
      3. tags — tag embeddings only, caption blanked, CFG
      4. gt — dataset images, bilinear-upscaled to the sampling resolution

    Sampling renders NATIVE 1024 regardless of training resolution — the
    point is to see the conditioning effect at full quality. The GT row is
    the dataset image bilinear-upscaled to 1024 for an honest comparison.
    All rows share one noise draw + schedule per sample, so row differences
    are pure conditioning. CFG uncond pass = blank caption + anchor tags.
    """
    model.eval()
    torch.cuda.set_device(dev)   # thread-local: allocations land on this GPU
    h = w = preview_res // 8
    b = len(pieces[0]["captions"])
    device = dev

    ctx_full, m_full = conditioner(pieces[0]["captions"])
    ctx_none, m_none = conditioner([""] * b)
    tag_ids = pieces[0]["tag_ids"].to(device)
    tag_mask = pieces[0]["tag_mask"].to(device)

    compression = 8
    x1 = (minres // (compression * patch)) ** 2
    x2 = (maxres // (compression * patch)) ** 2
    mu = mu_override if mu_override is not None else _mu_from_seq_len(
        h * w, x1, x2, mu_y1, mu_y2)

    gen = torch.Generator(device=device).manual_seed(seed)
    init_noise = torch.randn(b, 16, h, w, device=device, dtype=torch.bfloat16,
                             generator=gen)
    u_ids = torch.zeros_like(tag_ids)
    u_msk = torch.zeros_like(tag_mask)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    def sample_one_sample(j: int) -> torch.Tensor:
        """Render ONE sample's 3 sampling variants (batch-1 forwards).

        Within a GPU, samples render sequentially: a single 1024-res sample's
        activations are tiny next to the resident training state, so batch-1
        is the OOM-safe unit. (Parallelism lives ACROSS GPUs — the caller
        runs one thread per GPU.)
        """
        ctx_f, m_f = ctx_full[j:j + 1], m_full[j:j + 1]
        ctx_n, m_n = ctx_none[j:j + 1], m_none[j:j + 1]
        ids_f, msk_f = tag_ids[j:j + 1], tag_mask[j:j + 1]
        noise = init_noise[j:j + 1].clone()

        def sample(ctx, cmask, ids, msk):
            x = noise.clone()
            for i in range(steps):
                t = ts[i].expand(1)
                x_tok, pos, mask = prepare(x, ctx.shape[1], patch, cmask,
                                           taglen=ids.shape[1], tagmask=msk)
                x_tok_u, pos_u, mask_u = prepare(x, ctx_n.shape[1], patch,
                                                 m_n, taglen=ids.shape[1],
                                                 tagmask=u_msk[j:j + 1])
                with torch.autocast("cuda", torch.bfloat16):
                    v_c = model(x_tok, ctx, t, pos, mask, ids, msk)
                    v_u = model(x_tok_u, ctx_n, t, pos_u, mask_u, u_ids[j:j + 1], u_msk[j:j + 1])
                v = v_u + cfg_scale * (v_c - v_u)
                v = rearrange(v.float(), "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                              h=h // patch, w=w // patch, ph=patch, pw=patch)
                x = (x.float() + (ts[i + 1] - ts[i]) * v).to(torch.bfloat16)
            return x

        latents = [
            sample(ctx_f, m_f, ids_f, msk_f),                 # both
            sample(ctx_f, m_f, u_ids[j:j + 1], u_msk[j:j + 1]),   # text only
            sample(ctx_n, m_n, ids_f, msk_f),                 # tags only
        ]
        pixels = [vae_decode(ae, xl).clamp(-1, 1) for xl in latents]
        gt = pieces[0]["images"][j:j + 1].to(device, torch.bfloat16)
        gt = F.interpolate(gt, size=(h * 8, w * 8), mode="bilinear",
                           align_corners=False).clamp(-1, 1)
        pixels.append(gt)
        return torch.stack(pixels, dim=0).cpu()               # [4, 1, 3, R, R]

    outs = [sample_one_sample(j) for j in range(b)]
    stack = torch.cat(outs, dim=1)                            # [4, b, 3, R, R]
    model.train()
    return stack


def train(cfg: dict, config_path: str):
    _pin_sdpa_backends()
    n_gpus = int(cfg.get("n_gpus") or torch.cuda.device_count())
    devices = [f"cuda:{g}" for g in range(n_gpus)]
    driver = devices[0]

    from utils.tag_vocab import TagVocab
    vocab_path = cfg["tag_code"]["vocab_path"]
    vocab = TagVocab.load(vocab_path)
    vocab_size = len(vocab)
    code_fp = pq.read_table(cfg["tag_code"]["code_path"],
                            columns=["vocab_fingerprint"]
                            ).column("vocab_fingerprint")[0].as_py()

    bases, conditioners, aes, dit_cfg, enc_cfg = build_frozen(cfg, devices)
    patch = dit_cfg.patch
    # "context" (tags in the sequence) vs "kv" (tags as per-block keys/values,
    # never in the residual stream). One config field flips the whole forward;
    # the frozen bases never receive tag_ids, so the mode only affects the
    # trainable replicas built in make_train_factory.
    dit_cfg.tag_mode = str(cfg.get("tag_mode", "context"))
    print(f"  tag_mode: {dit_cfg.tag_mode}")

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    shutil.copy(config_path, os.path.join(cfg["ckpt_path"], os.path.basename(config_path)))

    dtype = torch.bfloat16
    resolution = int(cfg.get("resolution", 256))
    grad_accum = int(cfg.get("grad_accum", 1))
    uncond_ratio = float(cfg.get("uncond_ratio", 0.1))
    tag_drop_prob = float(cfg.get("tag_code", {}).get("tag_drop_prob", 0.1))
    minres, maxres = cfg.get("minres", 256), cfg.get("maxres", 1280)
    mu_y1 = cfg.get("mu_y1", 0.5)
    mu_y2 = cfg.get("mu_y2", 1.15)
    mu_sigma = cfg.get("mu_sigma", 1.0)
    mu_override = cfg.get("mu_override")
    max_steps = cfg.get("max_steps", 0)
    eval_interval = int(cfg.get("eval_interval", 50))
    save_every = int(cfg.get("save_every_n_steps", 200))
    master_seed = int(cfg.get("seed", 42))
    global_step = int(cfg.get("initial_global_step", 0))
    compression = 8

    # Resume: pick up the newest tagslora checkpoint unless the config says
    # otherwise. The save format (lora + tagcode + code buffers) is exactly
    # what MultiGPUWrapper._apply_state_dict expects via checkpoint_path.
    resume_path = cfg.get("resume_checkpoint")
    if resume_path is None:
        import glob as _glob
        import re as _re
        cands = (_glob.glob(os.path.join(cfg["ckpt_path"], "ckpts", "tagslora_step_*_ckpt.safetensors"))
                  or _glob.glob(os.path.join(cfg["ckpt_path"], "tagslora_step_*_ckpt.safetensors")))
        if cands:
            latest = max(cands, key=lambda p: int(_re.search(r"step_(\d+)_", p).group(1)))
            resume_path = latest
    if resume_path and not cfg.get("fresh_start"):
        n0 = int(__import__("re").search(r"step_(\d+)_", resume_path).group(1))
        if n0 > global_step:
            global_step = n0
        print(f"[resume] from {resume_path} (step {global_step})")

    lion_options = validate_lion_config(cfg)
    if lion_options is not None:
        def opt_factory(params):
            return Lion(params, **lion_options)
        opt_name = f"Lion {lion_options.get('betas')}"
    else:
        def opt_factory(params):
            return torch.optim.AdamW(params, lr=cfg.get("lr", 1e-4), betas=(0.9, 0.95))
        opt_name = "AdamW"
    lr = cfg.get("lr", 1e-4)
    warmup_steps = int(cfg.get("warmup", 100))

    wrapper = MultiGPUWrapper(
        model_factory=make_train_factory(cfg, dit_cfg, vocab_size,
                                         cfg["tag_code"]["code_path"], code_fp),
        optimizer_factory=opt_factory,
        scheduler_factory=lambda o: LinearLR(o, start_factor=1e-5, end_factor=1.0,
                                             total_iters=warmup_steps),
        n_gpus=n_gpus,
        checkpoint_path=(resume_path or ""),
        # The wrapper unconditionally casts every replica to `dtype` after
        # moving to device. bf16 keeps the frozen base at 25.6GB; trainable
        # masters are re-cast to fp32 below (the wrapper exposes .models as
        # public API exactly for this kind of tinkering).
        dtype=torch.bfloat16,
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
    )
    wrapper.setup()
    # Restore fp32 trainable masters (LoRA A/B + tagcode) post-cast on every
    # replica. setup() built optimizers against the Parameter OBJECTS, and
    # .data rebinding keeps identity — so ownership and optimizer wiring stay
    # intact; state was empty at setup, so no dtype conflicts.
    from krea2.model.lora import LoRALinear
    for g in range(n_gpus):
        m = wrapper.models[g]
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                mod.lora_A.data = mod.lora_A.data.float()
                mod.lora_B.data = mod.lora_B.data.float()
        if hasattr(m, "tagcode"):
            m.tagcode = m.tagcode.float()
        trainable_ps = [p for p in m.parameters() if p.requires_grad]
        assert all(p.dtype == torch.float32 for p in trainable_ps)
    print(f"  fp32 masters restored on {n_gpus} replica(s)")
    trainable = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable/1e6:.1f}M (LoRA + tag-code) | {opt_name} lr={lr:g}")

    x1_res = (minres // (compression * patch)) ** 2
    x2_res = (maxres // (compression * patch)) ** 2

    def fb_fn(gpu_id: int, model: SingleStreamDiT, images: torch.Tensor,
              captions: list[str], tag_ids: torch.Tensor,
              tag_mask: torch.Tensor) -> float:
        dev = devices[gpu_id]
        b = len(captions)
        is_uncond = [torch.rand(1).item() < uncond_ratio for _ in captions]
        dropped = ["" if u else c for u, c in zip(is_uncond, captions)]

        context, txtmask = conditioners[dev](dropped)
        txtlen = context.shape[1]

        with torch.no_grad():
            x0 = vae_encode(aes[dev], images)

        img_seq_len = (x0.shape[2] // patch) * (x0.shape[3] // patch)
        mu = _mu_from_seq_len(img_seq_len, x1_res, x2_res,
                              cfg.get("mu_y1", 0.5), cfg.get("mu_y2", 1.15))
        t = sample_timesteps(b, device=dev, mu=mu, sigma=cfg.get("mu_sigma", 1.0))

        # per-tag dropout: each tag token independently -> anchor id 0.
        # uncond samples lose the whole block (mask zeroed below).
        ids = tag_ids.clone()
        msk = tag_mask.clone()
        per_token = torch.rand(ids.shape, device=dev) < tag_drop_prob
        ids = ids.masked_fill(per_token, 0)
        msk = msk & ~torch.tensor(is_uncond, device=dev)[:, None]

        t4 = t[:, None, None, None].to(x0.dtype)
        noise = torch.randn(x0.shape, device=dev, dtype=x0.dtype)
        x_t = ((1.0 - t4) * x0 + t4 * noise).to(x0.dtype)
        taglen = ids.shape[1]
        x_tok, pos, mask = prepare(x_t, txtlen, patch, txtmask, taglen=taglen,
                                   tagmask=msk)

        with torch.autocast("cuda", dtype):
            v = model(x_tok, context, t, pos, mask, ids, msk)
        v_target = rearrange(noise - x0,
                             "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                             ph=patch, pw=patch)
        loss = F.mse_loss(v.float(), v_target.float())
        loss.backward()
        return loss.detach().item()

    wrapper.forward_backward_fn = fb_fn

    parquet_cfg = cfg.get("parquet_dataloader")
    global_batch = int(cfg.get("batch_size", 8)) * n_gpus * grad_accum
    dataset = ParquetTextImageDataset(
        batch_size=global_batch,
        parquet_sources=parquet_cfg["parquet_sources"],
        caption_columns=parquet_cfg["caption_columns"],
        filename_column=parquet_cfg.get("filename_column", "url"),
        width_column=parquet_cfg.get("width_column", "image_width"),
        height_column=parquet_cfg.get("height_column", "image_height"),
        loss_weight_column=parquet_cfg.get("loss_weight_column", None),
        image_folder_path=parquet_cfg.get("image_folder_path", ""),
        s3_image_source=parquet_cfg.get("s3_image_source", None),
        base_res=parquet_cfg.get("base_resolution", [resolution]),
        base_res_weights=parquet_cfg.get("base_resolution_weights", None),
        ratio_cutoff=parquet_cfg.get("ratio_cutoff", 2.0),
        resolution_step=parquet_cfg.get("resolution_step", 64),
        shuffle_tags=parquet_cfg.get("shuffle_tags", True),
        tag_drop_percentage=parquet_cfg.get("tag_drop_percentage", 0.0),
        uncond_percentage=0.0,
        seed=cfg.get("seed", 42),
        rank=0, num_gpus=1,
        offset=parquet_cfg.get("offset", 0),
        tokenizer=None, max_text_len=0,
        tag_column=parquet_cfg.get("tag_column"),
        tag_vocab_path=vocab_path if parquet_cfg.get("tag_column") else None,
        max_tags=cfg.get("tag_code", {}).get("max_tags", 128),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=parquet_cfg.get("num_workers", 2),
                        prefetch_factor=parquet_cfg.get("prefetch_factor", 2),
                        pin_memory=True, collate_fn=dataset.dummy_collate_fn)

    csv_path = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "a", newline="")
    wr = csv.writer(csv_file)
    if new:
        wr.writerow(["step", "loss", "lr", "time"])

    def save(tag):
        sd = {k: v.detach().cpu().contiguous()
              for k, v in wrapper.model.state_dict().items()
              if p_requires_grad(k, wrapper.model)}
        ckpt_dir = os.path.join(cfg["ckpt_path"], "ckpts")   # housekeep sweeps runs/*/ckpts
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"tagslora_step_{global_step}_{tag}.safetensors")
        save_file(sd, path)
        print(f"[ckpt] Saved {len(sd)} tensors -> {path}")

    def p_requires_grad(key, model):
        # state_dict keys -> params: lora + tagcode are the trainable set;
        # also keep the persistent code/fingerprint buffers for reload.
        return ("lora_A" in key or "lora_B" in key or "tagcode" in key)

    prompts = cfg.get("preview_prompts", [
        "1girl, solo, long hair, looking at viewer, blush, smile, outdoors",
        "a photo of a corgi riding a skateboard in a neon-lit city at night",
    ])

    torch.manual_seed(master_seed)
    stop = False
    pbar = tqdm(total=max_steps - global_step if max_steps else None, desc="tags-ddp")
    t0 = time.time()
    while not stop:
        torch.manual_seed(master_seed + global_step)
        wrapper.model.train()
        for batch_data in loader:
            batch_data = batch_data[0]
            images, captions = batch_data[0], batch_data[1]
            extras = batch_data[-1] if isinstance(batch_data[-1], dict) else None
            b = len(captions)
            usable = (b // n_gpus) * n_gpus
            if usable < n_gpus:
                continue
            images = images[:usable]
            captions = captions[:usable]
            per = usable // n_gpus
            tag_ids = extras["tag_ids"][:usable] if extras and "tag_ids" in extras else torch.zeros(usable, 1, dtype=torch.int64)
            tag_mask = extras["tag_mask"][:usable].bool() if extras and "tag_ids" in extras else torch.zeros(usable, 1, dtype=torch.bool)

            img_chunks = [c[0] for c in wrapper.split_batch(images)]
            chunks = []
            for g in range(n_gpus):
                s, e = g * per, (g + 1) * per
                chunks.append((img_chunks[g], captions[s:e],
                               tag_ids[s:e].to(f"cuda:{g}"),
                               tag_mask[s:e].to(f"cuda:{g}")))

            loss_sum = wrapper.step(chunks)
            loss_val = loss_sum / n_gpus
            global_step += 1
            pbar.update(1)

            lr_now = wrapper.scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_now:.2e}", step=global_step, res=resolution)
            wr.writerow([global_step, f"{loss_val:.6f}", lr_now, f"{time.time()-t0:.1f}"])
            if global_step % 50 == 0:
                csv_file.flush()

            if eval_interval and global_step % eval_interval == 0:
                # Dataset-driven 4-row preview. PARALLEL ACROSS GPUs (one
                # thread per GPU via the wrapper executor); WITHIN a GPU the
                # samples render one at a time (batch-1 forwards — the OOM-safe
                # unit next to the resident training state). 4 GPUs -> 4
                # samples in flight at once.
                prev_res = int(cfg.get("preview_res", 1024))
                prev_steps = int(cfg.get("preview_steps", 12))
                # Preview renders through the WRAPPER's own long-lived executor
                # threads (same threading proven by thousands of training
                # steps). Per thread: set_device (thread-local), torch.no_grad,
                # one sample at a time (batch-1 forwards are the OOM-safe unit
                # next to the resident training state). No empty_cache, no
                # backend swapping from worker threads.

                def render(g):
                    dev_g = devices[g]
                    piece = {
                        "images": images[g * per:(g + 1) * per][:2].cpu(),
                        "captions": captions[g * per:(g + 1) * per][:2],
                        "tag_ids": tag_ids[g * per:(g + 1) * per][:2],
                        "tag_mask": tag_mask[g * per:(g + 1) * per][:2],
                    }
                    try:
                        with torch.cuda.device(dev_g), torch.no_grad():
                            return preview_dataset(
                                wrapper.models[g], conditioners[dev_g],
                                aes[dev_g], [piece], patch, resolution,
                                prev_steps, None, dev_g,
                                cfg_scale=cfg.get("preview_cfg_scale", 4.5),
                                mu_y1=mu_y1, mu_y2=mu_y2, mu_sigma=mu_sigma,
                                minres=minres, maxres=maxres, mu_override=mu_override,
                                preview_res=prev_res, seed=master_seed + global_step)
                    finally:
                        wrapper.models[g].train()

                stacks = list(wrapper.executor.map(render, range(n_gpus)))
                grid_stack = torch.cat(stacks, dim=1)      # [4, n_gpus*2, 3, R, R]
                # make_grid wants 4D: flatten variant-major -> [4*cols, 3, R, R],
                # then nrow=cols lays out 4 rows of cols in exactly that order.
                flat = grid_stack.flatten(0, 1)
                grid = make_grid(flat, nrow=flat.shape[0] // 4,
                                 normalize=True, value_range=(-1, 1))
                arr = (grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0)
                       .to("cpu", torch.uint8).numpy())
                img_path = os.path.join(cfg["preview_path"], f"step_{global_step}.jpg")
                Image.fromarray(arr).save(img_path, quality=95)
                print(f"[preview] Saved {img_path} "
                      f"(rows: both | text | tags | gt, cols: {grid_stack.shape[1]})")
            if save_every and global_step % save_every == 0:
                save("ckpt")
            if max_steps and global_step >= max_steps:
                stop = True
                break

    if cfg.get("save_final", True):
        save("final")
    csv_file.close()
    print(f"Done at step {global_step}; peak VRAM: " + ", ".join(
        f"{d}={torch.cuda.max_memory_allocated(d)/2**30:.2f} GB" for d in devices))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = json.load(fh)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    train(cfg, args.config)


if __name__ == "__main__":
    main()
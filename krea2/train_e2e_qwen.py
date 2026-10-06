"""train_e2e_qwen.py — end-to-end Qwen3-VL + K2 DiT LoRA ("pseudo-VAE"), DDP.

Encoder: Qwen-VL(image) -> scene_graph JSON. Decoder: DiT(noise, Qwen(JSON), t).
The two halves share Qwen's weights; the training loop ALTERNATES objectives
per optimizer step (``objective_cycle``, default caption/diffusion 1:1):

- caption:   autoregressive CE on the JSON answer, image prefix, Qwen's native
             key-value extraction prompt (krea2/model/qwen_e2e.KIE_PROMPT).
- diffusion: flow-matching MSE with the DiT conditioned on Qwen(JSON text only);
             the gradient flows through the DiT into Qwen's LM LoRA.

Trainable: LoRA on the DiT (minus ``lora_exclude_prefixes``), the Qwen language
model and the Qwen vision tower (``qwen_lora_targets``); everything else is the
frozen bf16 base. ``lm_head`` is untouched (tied to the embeddings). Data
parallel via ramtorch MultiGPUWrapper (single process, NCCL, ZeRO-1). On
caption steps the DiT LoRA receives no grad; ``reduce_grads`` and Lion both
skip None grads, so its weights and momentum stay put.

Run:
    .venv/bin/python krea2/train_e2e_qwen.py krea2/configs/train_e2e_qwen.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration

from ramtorch.multi_gpu import MultiGPUWrapper

from krea2.lion import Lion, validate_lion_config
from krea2.model.autoencoder import QwenAutoencoder
from krea2.model.configs import MMDIT_CONFIGS
from krea2.model.lora import LoRALinear, inject_lora
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.qwen_e2e import E2EBatchBuilder, E2EModel, serialize_scene_graph
from krea2.model.sampling import prepare, sample as k2_sample
from krea2.train_utils import (
    _mu_from_seq_len, _pin_sdpa_backends, sample_timesteps, vae_encode,
)
from dataloaders.parquet_dataloader import ParquetTextImageDataset

torch.manual_seed(0)

GROUPS = ("dit", "qwen")


def _lora_params(module):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B


def make_factory(cfg: dict, dit_cfg, qwen_base):
    def factory() -> E2EModel:
        with torch.device("meta"):
            dit = SingleStreamDiT(dit_cfg)
        dit.load_state_dict(load_file(cfg["full_checkpoint"], device="cpu"),
                            strict=True, assign=True)
        inject_lora(dit, rank=int(cfg["lora_rank"]),
                    alpha=float(cfg.get("lora_alpha", cfg["lora_rank"])),
                    exclude_prefixes=tuple(cfg.get("lora_exclude_prefixes", ())))
        dit.requires_grad_(False)
        dit.grad_ckpt = True
        dit.txtfusion.grad_ckpt = True

        qwen = copy.deepcopy(qwen_base)
        qrank = int(cfg.get("qwen_lora_rank", cfg["lora_rank"]))
        inject_lora(qwen, rank=qrank, alpha=float(cfg.get("qwen_lora_alpha", qrank)),
                    include_substrings=tuple(cfg.get(
                        "qwen_lora_targets", ["model.language_model.", "model.visual."])))
        qwen.requires_grad_(False)
        qwen.config.use_cache = False
        qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

        for group, sub in (("dit", dit), ("qwen", qwen)):
            for p in _lora_params(sub):
                p.requires_grad_(True)
                p._e2e_group = group
        return E2EModel(dit, qwen)
    return factory


class _NoOpOpt:
    """Stands in for a torch optimizer on a GPU that owns no params."""
    param_groups: list = []

    def step(self):
        pass

    def zero_grad(self, set_to_none: bool = True):
        pass

    def state_dict(self):
        return {}


def train(cfg: dict, config_path: str):
    _pin_sdpa_backends()
    n_gpus = int(cfg.get("n_gpus") or torch.cuda.device_count())
    devices = [f"cuda:{g}" for g in range(n_gpus)]
    dit_cfg = MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")]
    patch = dit_cfg.patch
    compression = 8
    qwen_id = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    shutil.copy(config_path, os.path.join(cfg["ckpt_path"], os.path.basename(config_path)))

    resolution = int(cfg.get("resolution", 256))
    uncond_ratio = float(cfg.get("uncond_ratio", 0.1))
    minres, maxres = cfg.get("minres", 256), cfg.get("maxres", 1280)
    mu_y1, mu_y2 = cfg.get("mu_y1", 0.5), cfg.get("mu_y2", 1.15)
    mu_sigma = cfg.get("mu_sigma", 1.0)
    max_steps = cfg.get("max_steps", 0)
    eval_interval = int(cfg.get("eval_interval", 250))
    save_every = int(cfg.get("save_every_n_steps", 250))
    master_seed = int(cfg.get("seed", 42))
    global_step = int(cfg.get("initial_global_step", 0))
    cycle = list(cfg.get("objective_cycle", ["caption", "diffusion"]))
    assert cycle and all(o in ("caption", "diffusion") for o in cycle), cycle
    lrs = {"dit": float(cfg.get("lr_dit", 5e-5)), "qwen": float(cfg.get("lr_qwen", 2e-5))}
    warmup_steps = int(cfg.get("warmup", 100))

    resume_path = cfg.get("resume_checkpoint")
    if resume_path is None and not cfg.get("fresh_start"):
        cands = glob.glob(os.path.join(cfg["ckpt_path"], "ckpts", "e2e_step_*_*.safetensors"))
        if cands:
            resume_path = max(cands, key=lambda p: int(re.search(r"step_(\d+)_", p).group(1)))
    if resume_path:
        global_step = max(global_step, int(re.search(r"step_(\d+)_", resume_path).group(1)))
        print(f"[resume] from {resume_path} (step {global_step})")

    print("Loading batch builder + Qwen3-VL base + VAEs...")
    builder = E2EBatchBuilder(qwen_id, max_text_len=int(cfg.get("max_text_len", 1024)),
                              caption_max_pixels=int(cfg.get("caption_max_pixels", 512 * 512)),
                              caption_min_pixels=int(cfg.get("caption_min_pixels", 256 * 256)))
    qwen_base = Qwen3VLForConditionalGeneration.from_pretrained(qwen_id, dtype=torch.bfloat16)
    aes = {}
    for dev in devices:
        ae = QwenAutoencoder()
        ae.ae = ae.ae.to(torch.bfloat16)
        aes[dev] = ae.to(dev).eval().requires_grad_(False)

    lion_options = validate_lion_config(cfg)
    if lion_options is None:
        raise ValueError("train_e2e_qwen expects optimizer 'lion'")

    def opt_factory(params):
        groups = []
        for name in GROUPS:
            ps = [p for p in params if getattr(p, "_e2e_group", None) == name]
            if ps:
                groups.append({"params": ps, "lr": lrs[name], "name": name})
        if not groups:
            return _NoOpOpt()
        return Lion(groups, **lion_options)

    def sched_factory(opt):
        if isinstance(opt, _NoOpOpt):
            return None
        return LinearLR(opt, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps)

    wrapper = MultiGPUWrapper(
        model_factory=make_factory(cfg, dit_cfg, qwen_base),
        optimizer_factory=opt_factory,
        scheduler_factory=sched_factory,
        n_gpus=n_gpus,
        checkpoint_path=(resume_path or ""),
        # bf16 keeps the frozen bases small; LoRA masters go back to fp32 below.
        dtype=torch.bfloat16,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
    )
    wrapper.setup()
    del qwen_base
    # A None scheduler (no-op optimizer) would crash optimizer_step's loop.
    wrapper.schedulers = [s for s in wrapper.schedulers if s is not None] \
        if all(s is not None for s in wrapper.schedulers) else []

    # fp32 LoRA masters. .data rebinding keeps Parameter identity, so ZeRO-1
    # ownership and the optimizers built in setup() stay wired. A resume reads
    # the adapters again at full precision (the wrapper loaded them pre-cast).
    resume_sd = load_file(resume_path) if resume_path else {}
    for m in wrapper.models:
        for name, p in m.named_parameters():
            if p.requires_grad:
                src = resume_sd.get(name)
                p.data = (src.to(p.device, torch.float32) if src is not None
                          else p.data.float())
        bad = [n for n, p in m.named_parameters() if p.requires_grad and p.dtype != torch.float32]
        assert not bad, f"non-fp32 trainable masters: {bad[:3]}"
    del resume_sd
    if global_step:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # "scheduler.step() before optimizer.step()"
            for s in wrapper.schedulers:
                for _ in range(min(global_step, warmup_steps)):
                    s.step()
        print(f"  warmup fast-forwarded {min(global_step, warmup_steps)} steps "
              f"(Lion momentum restarts at zero)")

    named_groups = []   # per replica: [(group_label, param)] for grad-norm logging
    for m in wrapper.models:
        lst = []
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            label = ("dit" if name.startswith("dit.") else
                     "qwen_vis" if name.startswith("qwen.model.visual.") else "qwen_lm")
            lst.append((label, p))
        named_groups.append(lst)
    counts = {}
    for label, p in named_groups[0]:
        counts[label] = counts.get(label, 0) + p.numel()
    print("  Trainable: " + ", ".join(f"{k} {v/1e6:.1f}M" for k, v in counts.items())
          + f" | Lion {lion_options['betas']} lr_dit={lrs['dit']:g} lr_qwen={lrs['qwen']:g}")
    print(f"  objective cycle: {cycle}")

    trackio_run = bool(cfg.get("trackio_project"))
    if trackio_run:
        import trackio
        trackio.init(project=cfg["trackio_project"],
                     name=cfg.get("trackio_run") or os.path.basename(cfg["ckpt_path"].rstrip("/")),
                     config={"lr_dit": lrs["dit"], "lr_qwen": lrs["qwen"],
                             "warmup": warmup_steps, "resolution": resolution,
                             "batch_size": int(cfg.get("batch_size", 8)) * n_gpus,
                             "lora_rank": int(cfg["lora_rank"]),
                             "qwen_lora_rank": int(cfg.get("qwen_lora_rank", cfg["lora_rank"])),
                             "objective_cycle": cycle,
                             "base": os.path.basename(cfg["full_checkpoint"])})

    x1_res = (minres // (compression * patch)) ** 2
    x2_res = (maxres // (compression * patch)) ** 2
    step_stats = {}   # gpu_id -> per-step grad norms / timings, written by fb_fn

    def _grad_norms(gpu_id):
        sq = {}
        for label, p in named_groups[gpu_id]:
            if p.grad is not None:
                sq[label] = sq.get(label, 0.0) + float(p.grad.float().pow(2).sum())
        return {f"grad_norm_{k}": v ** 0.5 for k, v in sq.items()}

    def fb_fn(gpu_id: int, model: E2EModel, objective: str, payload: dict) -> float:
        dev = devices[gpu_id]
        torch.cuda.set_device(dev)
        t0 = time.time()
        if objective == "caption":
            enc = {k: v.to(dev, non_blocking=True) for k, v in payload.items()}
            with torch.autocast("cuda", torch.bfloat16):
                loss = model.caption_loss(**enc)
        else:
            images, ids, tmask = (payload["images"], payload["ids"].to(dev),
                                  payload["mask"].to(dev))
            with torch.no_grad():
                x0 = vae_encode(aes[dev], images)
            b = x0.shape[0]
            mu = _mu_from_seq_len((x0.shape[2] // patch) * (x0.shape[3] // patch),
                                  x1_res, x2_res, mu_y1, mu_y2)
            t = sample_timesteps(b, device=dev, mu=mu, sigma=mu_sigma)
            t4 = t[:, None, None, None].to(x0.dtype)
            noise = torch.randn(x0.shape, device=dev, dtype=x0.dtype)
            x_t = ((1.0 - t4) * x0 + t4 * noise).to(x0.dtype)
            with torch.autocast("cuda", torch.bfloat16):
                context, txtmask = model.encode_text(ids, tmask)
                x_tok, pos, mask = prepare(x_t, context.shape[1], patch, txtmask)
                v = model.dit(x_tok, context, t, pos, mask)
            v_target = rearrange(noise - x0, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                                 ph=patch, pw=patch)
            loss = F.mse_loss(v.float(), v_target.float())
        t_fwd = time.time() - t0
        loss.backward()
        entry = _grad_norms(gpu_id)
        entry["t_fwd"] = t_fwd
        entry["t_bwd"] = time.time() - t0 - t_fwd
        step_stats[gpu_id] = entry
        return loss.detach().item()

    wrapper.forward_backward_fn = fb_fn

    pcfg = cfg["parquet_dataloader"]
    global_batch = int(cfg.get("batch_size", 8)) * n_gpus
    dataset = ParquetTextImageDataset(
        batch_size=global_batch,
        parquet_sources=pcfg["parquet_sources"],
        caption_columns=pcfg["caption_columns"],
        filename_column=pcfg.get("filename_column", "url"),
        width_column=pcfg.get("width_column", "image_width"),
        height_column=pcfg.get("height_column", "image_height"),
        loss_weight_column=pcfg.get("loss_weight_column", None),
        image_folder_path=pcfg.get("image_folder_path", ""),
        s3_image_source=pcfg.get("s3_image_source", None),
        base_res=pcfg.get("base_resolution", [resolution]),
        base_res_weights=pcfg.get("base_resolution_weights", None),
        ratio_cutoff=pcfg.get("ratio_cutoff", 2.0),
        resolution_step=pcfg.get("resolution_step", 64),
        shuffle_tags=False,
        tag_drop_percentage=0.0,
        uncond_percentage=0.0,
        seed=cfg.get("seed", 42),
        rank=0, num_gpus=1,
        offset=pcfg.get("offset", 0),
        tokenizer=None, max_text_len=0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=pcfg.get("num_workers", 2),
                        prefetch_factor=pcfg.get("prefetch_factor", 2),
                        pin_memory=True, collate_fn=dataset.dummy_collate_fn)

    csv_path = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "a", newline="")
    wr = csv.writer(csv_file)
    if new:
        wr.writerow(["step", "objective", "loss", "lr_dit", "lr_qwen", "time"])

    def save(tag):
        sd = {k: v.detach().cpu().contiguous()
              for k, v in wrapper.model.state_dict().items()
              if ".lora_A" in k or ".lora_B" in k}
        ckpt_dir = os.path.join(cfg["ckpt_path"], "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"e2e_step_{global_step}_{tag}.safetensors")
        save_file(sd, path)
        print(f"[ckpt] Saved {len(sd)} tensors -> {path}")

    def current_lrs():
        opt = next((o for o in wrapper.optimizers if not isinstance(o, _NoOpOpt)), None)
        out = dict(lrs)
        if opt is not None:
            for pg in opt.param_groups:
                out[pg["name"]] = pg["lr"]
        return out

    # -- LR dial: live lr_dit / lr_qwen tuning through the config file --------
    # mtime-gated re-read (mangled JSON is warned and ignored), dial_grace-step
    # grace (reverting cancels), then a linear ramp of BOTH group lrs over
    # dial_ramp steps, applied by overriding param_group lr after each step.
    dial_grace = int(cfg.get("lr_dial", {}).get("grace_steps", 50))
    dial_ramp = int(cfg.get("lr_dial", {}).get("ramp_steps", 100))
    _dial = {"mtime": os.path.getmtime(config_path), "cur": dict(lrs), "target": None,
             "grace_left": 0, "ramp_from": None, "ramp_left": 0, "active": False}

    def _dial_apply(values):
        for s in wrapper.schedulers:
            s.base_lrs = [values[pg["name"]] for pg in s.optimizer.param_groups]
        for opt_ in wrapper.optimizers:
            for pg in opt_.param_groups:
                pg["lr"] = values[pg["name"]]

    def dial_tick():
        try:
            m = os.path.getmtime(config_path)
        except OSError as e:
            print(f"[lr-dial] config unreadable ({e}); keeping {_dial['cur']}")
            return
        if m != _dial["mtime"]:
            _dial["mtime"] = m
            try:
                with open(config_path) as f:
                    c = json.load(f)
                target = {"dit": float(c.get("lr_dit", _dial["cur"]["dit"])),
                          "qwen": float(c.get("lr_qwen", _dial["cur"]["qwen"]))}
            except Exception as e:
                print(f"[lr-dial] config change unreadable; keeping {_dial['cur']} ({e})")
                return
            if target != _dial["cur"] and target != _dial["target"]:
                _dial.update(target=target, grace_left=dial_grace,
                             ramp_from=dict(_dial["cur"]), ramp_left=dial_ramp, active=True)
                print(f"[lr-dial] {_dial['cur']} -> {target} armed: grace {dial_grace}, "
                      f"ramp {dial_ramp}")
            elif target == _dial["cur"] and _dial["target"] is not None:
                _dial.update(target=None, active=False)
                print("[lr-dial] reverted before it could ramp; cancelled")
        if _dial["target"] is None:
            return
        if _dial["grace_left"] > 0:
            _dial["grace_left"] -= 1
            return
        _dial["ramp_left"] -= 1
        frac = 1.0 - max(_dial["ramp_left"], 0) / dial_ramp
        _dial["cur"] = {k: _dial["ramp_from"][k] + (_dial["target"][k] - _dial["ramp_from"][k]) * frac
                        for k in GROUPS}
        _dial_apply(_dial["cur"])
        if _dial["ramp_left"] <= 0:
            print(f"[lr-dial] ramp complete: {_dial['cur']}")
            _dial["target"] = None

    def preview(images, jsons, per):
        """Rows: DiT(Qwen(GT JSON)) | DiT(Qwen(generated JSON)) | GT.
        Row 2 is the image -> JSON -> image round trip."""
        n_prev = int(cfg.get("preview_samples_per_gpu", 1))
        prev_res = int(cfg.get("preview_res", 512))
        sel = [(g, g * per + j) for g in range(n_gpus) for j in range(min(n_prev, per))]
        prompts = {g: [builder.caption_prompt(images[i]) for gg, i in sel if gg == g]
                   for g in range(n_gpus)}
        max_new = int(cfg.get("preview_max_new_tokens", 1536))

        def gen(g):
            dev = devices[g]
            torch.cuda.set_device(dev)
            q = wrapper.models[g].qwen
            q.eval()
            outs = []
            try:
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    for enc in prompts[g]:
                        enc = {k: v.to(dev) for k, v in enc.items()}
                        o = q.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                       use_cache=True)
                        outs.append(o[0, enc["input_ids"].shape[1]:].cpu())
            finally:
                q.train()
            return outs

        gen_ids = list(wrapper.executor.map(gen, range(n_gpus)))
        # Qwen pretty-prints; condition on the one-line training layout.
        gen_raw = {g: [builder.decode(x) for x in gen_ids[g]] for g in range(n_gpus)}
        gen_json = {g: [serialize_scene_graph(s) for s in gen_raw[g]] for g in range(n_gpus)}
        gt_json = {g: [jsons[i] for gg, i in sel if gg == g] for g in range(n_gpus)}
        pretok = {}
        for g in range(n_gpus):
            for s in gt_json[g] + gen_json[g] + [""]:
                if s not in pretok:
                    pretok[s] = builder.text([s])

        def render(g):
            dev = devices[g]
            torch.cuda.set_device(dev)
            m = wrapper.models[g]
            m.eval()

            def encoder(texts):
                ids = torch.cat([pretok[s][0] for s in texts]).to(dev)
                msk = torch.cat([pretok[s][1] for s in texts]).to(dev)
                return m.encode_text(ids, msk)

            cols = []
            try:
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    for j, i in enumerate([i for gg, i in sel if gg == g]):
                        row = []
                        for txt in (gt_json[g][j], gen_json[g][j]):
                            pil = k2_sample(m.dit, aes[dev], encoder, [txt], device=dev,
                                            width=prev_res, height=prev_res,
                                            steps=int(cfg.get("preview_steps", 28)),
                                            guidance=float(cfg.get("preview_cfg_scale", 4.5)),
                                            seed=master_seed + global_step + i,
                                            minres=minres, maxres=maxres, y1=mu_y1, y2=mu_y2)[0]
                            row.append(torch.from_numpy(np.asarray(pil)).permute(2, 0, 1)
                                       .float() / 127.5 - 1.0)
                        gt = F.interpolate(images[i:i + 1].float(), size=(prev_res, prev_res),
                                           mode="bilinear", align_corners=False)[0].clamp(-1, 1)
                        row.append(gt.cpu())
                        cols.append(torch.stack(row))      # [3, 3, R, R]
            finally:
                m.train()
            return cols

        cols = [c for cs in wrapper.executor.map(render, range(n_gpus)) for c in cs]
        stack = torch.stack(cols, dim=1)                    # [3 rows, n_cols, 3, R, R]
        grid = make_grid(stack.flatten(0, 1), nrow=stack.shape[1],
                         normalize=True, value_range=(-1, 1))
        arr = (grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0)
               .to("cpu", torch.uint8).numpy())
        img_path = os.path.join(cfg["preview_path"], f"step_{global_step}.jpg")
        Image.fromarray(arr).save(img_path, quality=95)
        txt_path = os.path.join(cfg["preview_path"], f"step_{global_step}.json")
        with open(txt_path, "w") as f:
            json.dump([{"gt": gt_json[g][j], "generated": gen_raw[g][j]}
                       for g in range(n_gpus) for j in range(len(gt_json[g]))],
                      f, ensure_ascii=False, indent=1)
        print(f"[preview] Saved {img_path} (rows: gt-json | generated-json | gt) + {txt_path}")
        if trackio_run:
            trackio.log({"preview": trackio.Image(
                arr, caption=f"step {global_step} — rows: gt-json | generated-json | gt"),
                "generated_json_0": gen_json[0][0][:2000] if gen_json[0] else ""},
                step=global_step)

    stop = False
    pbar = tqdm(total=max_steps - global_step if max_steps else None, desc="e2e-qwen")
    t0 = time.time()
    t_prev = time.time()
    while not stop:
        wrapper.model.train()
        for batch_data in loader:
            gap = time.time() - t_prev
            batch_data = batch_data[0]
            images, captions = batch_data[0], batch_data[1]
            usable = (len(captions) // n_gpus) * n_gpus
            if usable < n_gpus:
                continue
            per = usable // n_gpus
            images = images[:usable]
            jsons = [serialize_scene_graph(c) for c in captions[:usable]]
            torch.manual_seed(master_seed + global_step)
            objective = cycle[global_step % len(cycle)]

            chunks = []
            for g in range(n_gpus):
                s, e = g * per, (g + 1) * per
                if objective == "caption":
                    payload = builder.caption(images[s:e], jsons[s:e])
                else:
                    drop = (torch.rand(per) < uncond_ratio).tolist()
                    ids, msk = builder.text(["" if d else j for d, j in zip(drop, jsons[s:e])])
                    payload = {"images": images[s:e].to(devices[g], non_blocking=True),
                               "ids": ids, "mask": msk}
                chunks.append((objective, payload))

            t_step0 = time.time()
            loss_sum = wrapper.forward_backward_only(chunks)
            wrapper.reduce_grads()
            wrapper.clip_grads()
            wrapper.optimizer_step()
            step_total = time.time() - t_step0
            loss_val = loss_sum / n_gpus
            global_step += 1
            pbar.update(1)

            dial_tick()
            lr_now = current_lrs()
            pbar.set_postfix(obj=objective[:4], loss=f"{loss_val:.4f}", step=global_step)
            wr.writerow([global_step, objective, f"{loss_val:.6f}", lr_now["dit"],
                         lr_now["qwen"], f"{time.time()-t0:.1f}"])
            if global_step % 50 == 0:
                csv_file.flush()
            if trackio_run:
                metrics = {f"loss_{objective}": loss_val, "lr_dit": lr_now["dit"],
                           "lr_qwen": lr_now["qwen"], "gap": gap, "step_total": step_total}
                keys = {k for e in step_stats.values() for k in e}
                for k in keys:
                    vals = [e[k] for e in step_stats.values() if k in e]
                    metrics[f"{objective}/{k}" if k.startswith("grad_norm") else k] = \
                        sum(vals) / len(vals)
                trackio.log(metrics, step=global_step)
            step_stats.clear()

            if eval_interval and global_step % eval_interval == 0:
                preview(images, jsons, per)
            if save_every and global_step % save_every == 0:
                save("ckpt")
            t_prev = time.time()
            if max_steps and global_step >= max_steps:
                stop = True
                break

    if cfg.get("save_final", True):
        save("final")
    csv_file.close()
    if trackio_run:
        trackio.finish()
    print(f"Done at step {global_step}; peak VRAM: " + ", ".join(
        f"{d}={torch.cuda.max_memory_allocated(d)/2**30:.2f} GB" for d in devices))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-interval", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = json.load(fh)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.eval_interval is not None:
        cfg["eval_interval"] = args.eval_interval
    train(cfg, args.config)


if __name__ == "__main__":
    main()

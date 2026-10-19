"""Full supervised fine-tune of MOSS-Image (Base or SFT) on image + caption parquet data, torchrun.

Every transformer parameter is trainable, fp32 master weights, bf16 autocast compute. ``parallelism``:
  fsdp  FSDP2, sharded (one unit per block, one for the text projection, plus the root), fp32 reduce-scatter
  ddp   replicated (~110 GB/GPU static), grads all-reduced once per update (no_sync between micro-steps),
        fused AdamW
  zero  ZeRO-1 (utils/zero_sharding.py, ramtorch.MultiGPUWrapper's pattern): replicated weights, but every
        parameter is owned whole by one rank, which alone holds its optimizer state and EMA shadow. Once per
        update grads are reduced to their owners, owners step, then broadcast. Whole matrices, so Muon works.
        ``zero.clip`` "global" (default) | "local" (each owner clips its own grads; an approximation),
        ``zero.comm_dtype`` (grads only), ``zero.mode`` "coalesced" | "async" | "loop", ``zero.bucket_mb``.
        moss/tools/check_zero_sync.py; comms measured by moss/tools/probe_comms.py.
The Qwen3.5 text encoder and the FLUX.2 VAE are frozen and replicated on every rank.

Throughput switches: ``batched_forward`` (one padded pass per micro-batch instead of the release's per-sample
loop; moss/tools/check_batched_forward.py), ``seq_pad`` (pad the text so image + text is a multiple of it: few
static shapes), ``batched_text`` (one right-padded encoder pass), ``encode_update_text`` (one encoder pass over
the whole update's captions), ``grad_checkpoint`` (true/"all" | false/"none" | N = the first N blocks),
``compile`` (regional torch.compile of every block, ``compile_dynamic`` static|auto|dynamic), ``sdpa`` "cudnn"
(masked SDPA otherwise runs the ~2x slower "efficient" kernel). moss/tools/probe_sft_throughput.py picks them.

Recipe switches (moss/tools/check_sft_recipe.py): ``optimizer`` "adamw" | "muon" (hybrid Muon-AdamW as in the
MOSS-Image report, DDP only; utils/muon.py), ``timestep.kind`` "moss_final" (the report's last pretraining
interval), ``ema.decay`` (fp32 EMA; previews add an EMA column and the ``transformer/`` export IS the EMA),
``qk_norm_decay_to_one``.

Objective: rectified-flow velocity matching in MOSS's convention, x_t = (1 - t) x0 + t eps, v = eps - x0,
timestep = t * 1000; t = shift(sigmoid(N(mean, std))).

Data: dataloaders/parquet_dataloader.ParquetTextImageDataset, the krea2 fine-tune loader: aspect-ratio buckets,
one caption column drawn per row by weight (over the columns that row has non-empty), tag columns shuffled,
uncond dropout, images fetched by url. Every global batch shares one bucket; rank k takes slice k. JSON
captions (scene_graph) are re-serialized compactly, the form MOSS's prompt compiler emits.

torchrun --nproc_per_node 2 moss/train_sft.py CFG [--resume RUN/checkpoints/update-XXXXXX | --init-from CKPT]

``--resume`` continues a run exactly (same config identity and world size). ``--init-from`` starts a new run
from any of these checkpoints' model + AdamW state (FSDP or DDP; DCP reshards), with fresh data, RNG and
update counter, so batch size and parallelism may change.

Stop cleanly with ``touch RUN/STOP`` (or one SIGINT/SIGTERM): finish the update, save, exit.
Checkpoints: a sharded torch.distributed.checkpoint of model + AdamW, per-rank RNG and the data position; at
milestones and at the end also ``transformer/``, a bf16 release-layout folder (same keys and config as the
release) that ``load_transformer`` and the release pipeline load directly.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import contextlib
import json
import math
import random
import shutil
import signal
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict, get_state_dict,
                                                     set_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from moss.model.sampling import cfg_euler, moss_sigmas
from moss.train_utils import JsonLog, finite, load_validation_prompts, preview_grid, restore_rng, rng_state

IDENTITY_KEYS = ("seed", "model_path", "data", "per_rank_batch", "grad_accum", "timestep")
# ZeRO ranks hold different optimizers / param groups, so the default list-indexed ``param_groups`` keys
# would collide across ranks in the checkpoint; flattened, every key is per parameter name.
ZERO_STATE_OPTIONS = StateDictOptions(flatten_optimizer_state_dict=True)


def validate_config(cfg):
    for key in ("max_updates", "eval_interval", "save_interval", "milestone_interval", "keep_checkpoints",
                "per_rank_batch", "grad_accum"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg.get("parallelism", "fsdp") not in ("fsdp", "ddp", "zero"):
        raise ValueError("parallelism must be 'fsdp', 'ddp' or 'zero'")
    if cfg.get("optimizer", "adamw") not in ("adamw", "muon"):
        raise ValueError("optimizer must be 'adamw' or 'muon'")
    if cfg.get("optimizer") == "muon" and cfg.get("parallelism", "fsdp") == "fsdp":
        raise ValueError("muon orthogonalizes whole matrices: it needs parallelism 'ddp' or 'zero'")
    z = cfg.get("zero", {})
    if z.get("clip", "global") not in ("global", "local"):
        raise ValueError("zero.clip must be 'global' or 'local'")
    if z.get("mode", "coalesced") not in ("coalesced", "async", "loop"):
        raise ValueError("zero.mode must be 'coalesced', 'async' or 'loop'")
    if z.get("comm_dtype") not in (None, "float32", "bfloat16"):
        raise ValueError("zero.comm_dtype must be null, 'float32' or 'bfloat16'")
    if cfg.get("muon", {}).get("state_dtype") not in (None, "float32", "bfloat16"):
        raise ValueError("muon.state_dtype must be null, 'float32' or 'bfloat16'")
    if cfg.get("ema") and not 0 < cfg["ema"]["decay"] < 1:
        raise ValueError("ema.decay must be in (0, 1)")
    checkpoint_count(cfg.get("grad_checkpoint", True), 42)
    if type(cfg.get("warmup", 0)) is not int or cfg.get("warmup", 0) < 0:
        raise ValueError("warmup must be a nonnegative integer")
    d = cfg["data"]
    if d.get("resolution_step", 64) % 16 or any(r % 16 for r in d["base_resolution"]):
        raise ValueError("base_resolution and resolution_step must be multiples of 16 (VAE /8, 2x2 patchify)")
    v = cfg["validation"]
    if any(v[k] % 16 for k in ("width", "height")):
        raise ValueError("validation width/height must be multiples of 16")
    if len(cfg["validation_prompts"]) != len(cfg["validation_seeds"]):
        raise ValueError("validation_prompts and validation_seeds differ in length")


def checkpoint_count(value, layers):
    if value is True or value == "all":
        return layers
    if value is False or value == "none":
        return 0
    if type(value) is int and 0 <= value <= layers:
        return value
    raise ValueError(f"grad_checkpoint must be true/false, 'all'/'none' or 0..{layers}, got {value!r}")


def block_call(layer, *args):
    if layer.checkpointed and torch.is_grad_enabled():
        return checkpoint(layer, *args, use_reentrant=False)
    return layer(*args)


def configure(model, cfg):
    """Throughput switches on a (not yet wrapped) MOSSImageTransformer."""
    model.batched = cfg.get("batched_forward", False)
    model.text_pad = cfg.get("text_pad", 64)
    model.seq_pad = cfg.get("seq_pad")
    count = checkpoint_count(cfg.get("grad_checkpoint", True), len(model.layers))
    for index, layer in enumerate(model.layers):
        layer.checkpointed = index < count
    model.block_call = block_call
    from moss.model import transformer
    if cfg.get("sdpa", "default") == "cudnn":
        from torch.nn.attention import SDPBackend
        transformer.SDPA_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                                     SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    else:
        transformer.SDPA_BACKENDS = None
    if cfg.get("compile", False):
        import torch._dynamo
        import torch._inductor.config
        # The mixed row/column reduction it fuses for the RMSNorm backward needs >232 KB of shared memory at
        # B x S >= ~4.6k tokens ("No valid triton configs"); torch 2.10.
        torch._inductor.config.triton.mix_order_reduction = False
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit,
                                                                2048)
        # "auto": static first, then only the dims that actually vary (sequence length) go symbolic.
        dynamic = {"auto": None, "static": False, "dynamic": True}[cfg.get("compile_dynamic", "auto")]
        # Compile forward only, so wrapper hooks (FSDP) and checkpointing stay outside the graph. Not the two
        # text blocks: short sequences, and under seq_pad their length differs per bucket (a recompile each).
        for block in model.layers:
            block.forward = torch.compile(block.forward, dynamic=dynamic)
    return model


EMBEDDINGS = ("video_embedder.", "text_embedder.proj.")
MUON_SPLITS = {"proj.ffn_up_gate.weight": 2, "qkv.weight": 3}  # fused rows: [hidden; gate], [q; k; v]


def muon_eligible(name, p):
    """The report's split: Muon for the hidden matrices of the 42 blocks and the 2 text blocks; AdamW for the
    embeddings, the output head, time / modulation MLPs, norms and biases."""
    return p.ndim == 2 and name.endswith(".weight") and (
        (name.startswith("layers.") and ".proj." in name) or name.startswith("text_embedder.blocks."))


def build_optimizers(module, cfg, owned=None):
    """``optimizer`` "adamw" (default: one AdamW over everything at ``weight_decay``) or "muon" (hybrid
    Muon-AdamW, DDP / ZeRO only). Muon-AdamW: weight decay on every matrix but the embeddings; none on 1-D
    tensors. ``owned`` (a set of names, ZeRO): only those parameters; empty groups / optimizers are dropped."""
    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad and (owned is None or n in owned)]
    fused = cfg.get("parallelism", "fsdp") in ("ddp", "zero")
    adam = dict(lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg.get("eps", 1e-8), fused=fused or None)
    if cfg.get("optimizer", "adamw") == "adamw":
        return [torch.optim.AdamW([p for _, p in named], weight_decay=cfg["weight_decay"], **adam)] if named else []
    from utils.muon import Muon
    m = cfg.get("muon", {})
    by_split = {}
    for n, p in named:
        if muon_eligible(n, p):
            parts = next((k for suffix, k in MUON_SPLITS.items() if n.endswith(suffix)), 1)
            by_split.setdefault(parts, []).append(p)
    muon_groups = [{"params": ps, "splits": None if k == 1 else [ps[0].shape[0] // k] * k}
                   for k, ps in sorted(by_split.items())]
    optimizers = []
    if muon_groups:
        state_dtype = getattr(torch, m["state_dtype"]) if m.get("state_dtype") else None
        optimizers.append(Muon(muon_groups, lr=cfg["lr"], momentum=m.get("momentum", 0.95),
                               nesterov=m.get("nesterov", True), weight_decay=cfg["weight_decay"],
                               ns_steps=m.get("ns_steps", 5), rms_scale=m.get("rms_scale", 0.2),
                               batch=m.get("batch", 16), state_dtype=state_dtype))
    rest = [(n, p) for n, p in named if not muon_eligible(n, p)]
    decay = [p for n, p in rest if p.ndim == 2 and not n.startswith(EMBEDDINGS)]
    plain = [p for n, p in rest if not (p.ndim == 2 and not n.startswith(EMBEDDINGS))]
    groups = [g for g in ({"params": decay, "weight_decay": cfg["weight_decay"]},
                          {"params": plain, "weight_decay": 0.0}) if g["params"]]
    if groups:
        optimizers.append(torch.optim.AdamW(groups, **adam))
    return optimizers


class EMA:
    """fp32 shadow of every trainable parameter (local shards under FSDP; only ``names``, the owned ones,
    under ZeRO), decay warmed up as min(decay, (1 + n) / (10 + n)) so early updates are not pinned to the
    start weights."""

    def __init__(self, module, decay, names=None):
        self.decay = decay
        self.names = [n for n, p in module.named_parameters() if p.requires_grad and (names is None or n in names)]
        params = dict(module.named_parameters())
        self.shadow = {n: params[n].detach().float().clone() for n in self.names}
        self.count = 0

    @torch.no_grad()
    def update(self, module):
        self.count += 1
        decay = min(self.decay, (1 + self.count) / (10 + self.count))
        if not self.names:
            return
        params = dict(module.named_parameters())
        torch._foreach_lerp_([self.shadow[n] for n in self.names], [params[n].detach().float() for n in self.names],
                             1 - decay)

    @torch.no_grad()
    def swap(self, module):
        """Exchange the live weights and the shadow in place (call twice to restore)."""
        params = dict(module.named_parameters())
        for n in self.names:
            p, s = params[n], self.shadow[n]
            live = p.detach().float().clone()
            p.copy_(s)
            s.copy_(live)

    def state_dict(self):
        return {"shadow": self.shadow, "count": torch.tensor(self.count)}

    def load_state_dict(self, state):
        self.count = int(state["count"])


@contextlib.contextmanager
def swapped(ema, module):
    ema.swap(module)
    try:
        yield
    finally:
        ema.swap(module)


def sample_t(n, timestep, device):
    """``kind`` "shifted_logit_normal" (default): sigmoid(N(mean, std)), then MOSS's schedule shift.
    "moss_final": the last pretraining interval of the MOSS-Image report (eq. 4, Table 6): S = 1 - sigmoid(Z),
    Z ~ N(mean, std^2), truncated to S <= cutoff and rescaled by 1/cutoff onto the grid {1..steps}/steps, with
    weight ``uniform`` of discrete-uniform grid points; no shift."""
    if timestep.get("kind", "shifted_logit_normal") == "shifted_logit_normal":
        u = torch.sigmoid(torch.randn(n, device=device) * timestep["std"] + timestep["mean"])
        s = timestep["shift"]
        return s * u / (1 + (s - 1) * u)
    steps, a, mean, std = timestep.get("steps", 1000), timestep["cutoff"], timestep["mean"], timestep["std"]
    # S <= a  <=>  Z >= log((1 - a) / a): sample the truncated normal by inverse CDF.
    low = torch.special.ndtr(torch.tensor((math.log((1 - a) / a) - mean) / std, dtype=torch.float64)).item()
    u = (low + (1 - low) * torch.rand(n, device=device, dtype=torch.float64)).clamp(max=1 - 1e-12)
    sigma = (1 - torch.sigmoid(mean + std * torch.special.ndtri(u))) / a
    index = torch.ceil(sigma * steps).clamp(1, steps)
    uniform = torch.randint(1, steps + 1, (n,), device=device, dtype=torch.float64)
    index = torch.where(torch.rand(n, device=device) < timestep.get("uniform", 0.0), uniform, index)
    return (index / steps).float()


class BatchStream:
    """This rank's ParquetTextImageDataset batches, resumable at (epoch, position)."""

    def __init__(self, cfg, rank, world):
        from dataloaders.parquet_dataloader import ParquetTextImageDataset
        d = self.d = cfg["data"]
        self.dataset = ParquetTextImageDataset(
            batch_size=world * cfg["per_rank_batch"], parquet_sources=d["parquet_sources"],
            caption_columns=d["caption_columns"], filename_column=d.get("filename_column", "url"),
            width_column=d.get("width_column", "image_width"), height_column=d.get("height_column", "image_height"),
            base_res=d["base_resolution"], base_res_weights=d.get("base_resolution_weights"),
            ratio_cutoff=d.get("ratio_cutoff", 2.0), resolution_step=d.get("resolution_step", 64),
            shuffle_tags=d.get("shuffle_tags", True), tag_drop_percentage=d.get("tag_drop_percentage", 0.0),
            uncond_percentage=d.get("uncond_percentage", 0.1), seed=cfg["seed"], rank=rank, num_gpus=world,
            timeout=d.get("timeout", 30), offset=0, tokenizer=None, max_text_len=0, min_pixels=d.get("min_pixels"),
            s3_image_source=d.get("s3_image_source"))
        self.full = self.dataset.batches
        self.epoch, self.position, self._iter = 0, 0, None

    def __len__(self):
        return len(self.full)

    def _start(self):
        from torch.utils.data import DataLoader
        self.dataset.batches = self.full[self.position:]
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=self.d.get("num_workers", 4),
                            prefetch_factor=self.d.get("prefetch_factor", 4), pin_memory=True,
                            collate_fn=self.dataset.dummy_collate_fn)
        self._iter = iter(loader)

    def next(self):
        """(pixels [B, 3, H, W] in [-1, 1], captions, wait seconds)."""
        from dataloaders.caption_stream import format_caption
        started = time.monotonic()
        while True:
            if self._iter is None:
                self._start()
            try:
                item = next(self._iter)[0]
                break
            except StopIteration:
                self._iter = None
                self.epoch, self.position = self.epoch + 1, 0
                self.dataset.resample()
                self.full = self.dataset.batches
        self.position += 1
        captions = [format_caption(c, "compact_json") if c.lstrip().startswith("{") else c for c in item[1]]
        return item[0], captions, time.monotonic() - started

    def state_dict(self):
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        if self._iter is not None:
            raise RuntimeError("load_state_dict must precede the first next()")
        for _ in range(state["epoch"]):
            self.dataset.resample()
        self.full = self.dataset.batches
        self.epoch, self.position = state["epoch"], state["position"]


class FineTune:
    def __init__(self, cfg, rank, world, device, mesh, cpu_group=None):
        from moss.model.conditioners import Decoder, Encoder, TextEncoder
        from moss.model.transformer import load_transformer
        self.cfg, self.rank, self.world, self.device, self.cpu_group = cfg, rank, world, device, cpu_group
        self.ddp = cfg.get("parallelism", "fsdp") == "ddp"
        self.zero = None
        model = load_transformer(Path(cfg["model_path"]) / "transformer", device, dtype=torch.float32)
        model.requires_grad_(True).train()
        configure(model, cfg)
        self.module = model
        if cfg.get("parallelism") == "zero":
            from utils.zero_sharding import ZeroSync
            z = cfg.get("zero", {})
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            groups = None
            if cfg.get("optimizer") == "muon":
                muon = [i for i, (n, p) in enumerate(named) if muon_eligible(n, p)]
                groups = [muon, sorted(set(range(len(named))) - set(muon))]
            self.zero = ZeroSync(named, rank, world, groups=groups, mode=z.get("mode", "coalesced"),
                                 comm_dtype=getattr(torch, z["comm_dtype"]) if z.get("comm_dtype") else None,
                                 bucket_mb=z.get("bucket_mb", 1024))
            self.model = model
        elif self.ddp:
            self.model = DistributedDataParallel(model, device_ids=[device.index], gradient_as_bucket_view=True,
                                                 bucket_cap_mb=cfg.get("ddp_bucket_mb", 256))
        else:
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            # The per-sample loop would re-gather a block resharded after each call once per sample.
            fully_shard(model.text_embedder, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
            for layer in model.layers:
                fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
            fully_shard(model, mesh=mesh, mp_policy=policy)
            self.model = model
        self.channels = model.config.latent_channels
        self.encoder = TextEncoder(cfg["model_path"], device, batched=cfg.get("batched_text", False))
        self.vae_encoder = Encoder(cfg["model_path"], device)
        self.decoder = Decoder(cfg["model_path"], device, vae=self.vae_encoder.vae)
        self.replicated = self.ddp or self.zero is not None
        self.state_options = ZERO_STATE_OPTIONS if self.zero is not None else None
        self.ema = EMA(model, cfg["ema"]["decay"], self.owned_names()) if cfg.get("ema") else None

    def owned_names(self):
        """ZeRO: the parameters whose optimizer state / EMA this rank holds; otherwise None (all)."""
        return set(self.zero.owned_names()) if self.zero is not None else None

    def ema_weights(self):
        """Context in which every replica runs with the EMA weights (collective under ZeRO)."""
        if self.zero is not None:
            return self.zero.swapped(self.ema, self.module)
        return swapped(self.ema, self.module)

    def sync_context(self, last):
        """DDP: skip the all-reduce on every micro-step but the last."""
        return contextlib.nullcontext() if last or not self.ddp else self.model.no_sync()

    def reshard(self):
        for module in self.module.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

    def loss(self, pixels, captions, contexts=None):
        x0 = self.vae_encoder(pixels.to(self.device, non_blocking=True))
        contexts = self.encoder(captions) if contexts is None else contexts
        t = sample_t(len(x0), self.cfg["timestep"], self.device)
        eps = torch.randn_like(x0)
        tt = t.view(-1, 1, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = self.model((1 - tt) * x0 + tt * eps, t * 1000, contexts).float()
        return F.mse_loss(v, eps - x0), t

    @torch.no_grad()
    def evaluate(self, update, log):
        """Collective. FSDP: every rank samples every prompt (each forward gathers shards). DDP: rank r samples
        prompts r, r + world, ... on its own replica and rank 0 gathers them. Rank 0 decodes and writes the grid."""
        from moss.model.conditioners import to_pil
        cfg, v = self.cfg, self.cfg["validation"]
        folder = Path(cfg["output_dir"]) / "previews"
        base_path = folder / "base.pt"
        key = {"prompts": cfg["validation_prompts"], "seeds": cfg["validation_seeds"], "validation": v,
               "model_path": cfg["model_path"]}
        base = None
        if base_path.exists():
            saved = torch.load(base_path, map_location="cpu")
            base = saved["latents"] if saved["key"] == key else None
        uncond = self.encoder([cfg.get("negative_prompt", "")])
        sigmas = moss_sigmas(v["steps"], v.get("shift", 5.0))

        def render():
            jobs = list(enumerate(zip(cfg["validation_prompts"], cfg["validation_seeds"])))
            if self.replicated:
                jobs = jobs[self.rank::self.world]
            mine = {}
            for index, (prompt, seed) in jobs:
                contexts = self.encoder([prompt])
                generator = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn((1, self.channels, v["height"] // 16, v["width"] // 16), generator=generator,
                                    device=self.device)
                mine[index] = cfg_euler(self.module, noise, sigmas, contexts, uncond, v["guidance"],
                                        tuple(v.get("uncond_skip", (22,)))).cpu()
            if self.replicated:
                parts = [None] * self.world
                dist.all_gather_object(parts, mine, group=self.cpu_group)
                mine = {k: x for part in parts for k, x in part.items()}
            else:
                self.reshard()
            return [mine[i] for i in range(len(cfg["validation_prompts"]))]

        columns, labels = [render()], [f"SFT @ {update}"]
        if self.ema is not None and update > 0:
            with self.ema_weights():
                columns.append(render())
            labels.append(f"EMA @ {update}")
        if self.rank != 0:
            return
        folder.mkdir(parents=True, exist_ok=True)
        if base is None and update == 0:
            torch.save({"key": key, "latents": columns[0]}, base_path)
            base = None
        if base is not None:
            columns, labels = [base, *columns], ["base", *labels]
        rows = [[to_pil(self.decoder(col[i].to(self.device)))[0] for col in columns]
                for i in range(len(cfg["validation_prompts"]))]
        path = folder / f"validation-{update:06d}.jpg"
        preview_grid(rows, labels, path, cell=512)
        log(event="eval", update=update, path=str(path))
        print(f"[sft] previews {path}", flush=True)


def export_transformer(full, model_path, folder):
    """bf16 state in the release ``transformer/`` layout (one shard + index + the release config)."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    release = Path(model_path) / "transformer"
    weight_map = json.loads((release / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(release / weight_map["rope.bands"]), "pt") as f:
        full = {**full, "rope.bands": f.get_tensor("rope.bands")}
    folder.mkdir(parents=True, exist_ok=True)
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(full, str(folder / shard))
    total = sum(v.numel() * v.element_size() for v in full.values())
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": {k: shard for k in sorted(full)}}, indent=2))
    shutil.copy(Path(model_path) / "transformer" / "config.json", folder / "config.json")


class Saver:
    """Checkpoints that overlap training; ``finish`` publishes the pending one.

    ``start`` (collective) hands the sharded model + AdamW state to ``dcp.async_save``; an export gathers the
    bf16 transformer to rank 0's CPU and writes it from a thread. A directory is renamed into place only once
    every rank's write has landed. Waits use gloo groups so a rank blocked on disk idles instead of spinning.
    """

    def __init__(self, run, cfg, cpu_group):
        self.run, self.cfg, self.group, self.pending = run, cfg, cpu_group, None
        # async_save finishes with collectives from its own thread; it must not share poll()'s group.
        self.save_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))

    def start(self, optimizer, stream, update, elapsed, export):
        self.finish()
        run, cfg = self.run, self.cfg
        root = Path(cfg["output_dir"]) / "checkpoints"
        name = f"update-{update:06d}"
        temporary, destination = root / f".{name}-partial", root / name
        if run.rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
        dist.barrier(group=self.group)
        model_state, optim_state = get_state_dict(run.model, optimizer, options=run.state_options)
        state = {"model": model_state, "optim": optim_state}
        if run.ema is not None:
            state["ema"] = run.ema.state_dict()
        future = dcp.async_save(state, checkpoint_id=str(temporary / "dcp"), process_group=self.save_group)
        torch.save(rng_state(), temporary / f"rng-rank{run.rank}.pt")
        writer = None
        if export:
            # The export is the EMA when there is one (the release's own checkpoints are EMA weights).
            if run.zero is not None:
                # Owners swap their EMA shard in and broadcast, so rank 0's replica holds the whole EMA.
                with run.ema_weights() if run.ema is not None else contextlib.nullcontext():
                    full = ({k: v.to("cpu", torch.bfloat16) for k, v in run.module.state_dict().items()}
                            if run.rank == 0 else {})
            elif run.ddp:
                full = run.module.state_dict() if run.rank == 0 else {}
                if run.ema is not None and run.rank == 0:
                    full.update({n: s.cpu() for n, s in run.ema.shadow.items()})
            else:
                full = get_model_state_dict(run.model, options=StateDictOptions(full_state_dict=True,
                                                                                cpu_offload=True))
                if run.ema is not None:
                    shadow = {n: s.full_tensor().cpu() for n, s in run.ema.shadow.items()}
                    full.update(shadow if run.rank == 0 else {})
            if run.rank == 0:
                full = {k: v.to(torch.bfloat16).contiguous() for k, v in full.items()}
                writer = threading.Thread(target=export_transformer,
                                          args=(full, cfg["model_path"], temporary / "transformer"))
                writer.start()
            del full
        if run.rank == 0:
            torch.save({"stream": stream.state_dict(), "update": update, "elapsed": elapsed, "world": run.world,
                        "identity": {k: cfg[k] for k in IDENTITY_KEYS}}, temporary / "trainer.pt")
            (temporary / "metadata.json").write_text(json.dumps(
                {"update": update, "elapsed_seconds": elapsed, "model_path": cfg["model_path"],
                 "full_finetune": True, "has_transformer_export": export, "stream": stream.state_dict()}, indent=2))
        self.pending = (future, writer, temporary, destination)
        if run.rank == 0:
            print(f"[sft] checkpoint {name} staged; writing in the background", flush=True)

    def poll(self):
        """Collective, cheap: publish the pending checkpoint once every rank's writes are done."""
        if self.pending is None:
            return
        future, writer = self.pending[:2]
        done = torch.tensor([float(future.done() and (writer is None or not writer.is_alive()))])
        dist.all_reduce(done, op=dist.ReduceOp.MIN, group=self.group)
        if done.item():
            self.finish()

    def finish(self):
        if self.pending is None:
            return
        future, writer, temporary, destination = self.pending
        self.pending = None
        future.result()
        if writer is not None:
            writer.join()
        dist.barrier(group=self.group)
        if self.run.rank == 0:
            if destination.exists():
                shutil.rmtree(destination)
            temporary.rename(destination)
            kept = sorted(p for p in destination.parent.glob("update-*") if p.is_dir())
            for old in kept[:-self.cfg["keep_checkpoints"]]:
                if int(old.name.split("-")[1]) % self.cfg["milestone_interval"] or not (old / "transformer").exists():
                    shutil.rmtree(old)
                elif (old / "dcp").exists():
                    # Milestones keep the bf16 export; the ~60 GB resume shards go.
                    shutil.rmtree(old / "dcp")
            print(f"[sft] checkpoint {destination} published", flush=True)
        dist.barrier(group=self.group)


def load(path, run, optimizer, stream, cfg):
    path = Path(path)
    trainer = torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)
    if trainer["identity"] != {k: cfg[k] for k in IDENTITY_KEYS} or trainer["world"] != run.world:
        raise ValueError("Resume config/world size differs from the checkpoint's training identity")
    model_state, optim_state = get_state_dict(run.model, optimizer, options=run.state_options)
    state = {"model": model_state, "optim": optim_state}
    if run.ema is not None:
        state["ema"] = run.ema.state_dict()
    dcp.load(state, checkpoint_id=str(path / "dcp"))
    set_state_dict(run.model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"],
                   options=run.state_options)
    if run.ema is not None:
        run.ema.load_state_dict(state["ema"])
    stream.load_state_dict(trainer["stream"])
    restore_rng(torch.load(path / f"rng-rank{run.rank}.pt", map_location="cpu", weights_only=False))
    return trainer["update"], trainer["elapsed"]


def init_from(path, run, optimizer):
    """Model + optimizer state of any checkpoint here (FSDP or DDP; the optimizer must match); data, RNG, the
    update counter and the EMA (re-seeded from the loaded weights) start fresh."""
    path = Path(path)
    model_state, optim_state = get_state_dict(run.model, optimizer, options=run.state_options)
    state = {"model": model_state, "optim": optim_state}
    dcp.load(state, checkpoint_id=str(path / "dcp"))
    set_state_dict(run.model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"],
                   options=run.state_options)
    if run.ema is not None:
        run.ema = EMA(run.module, run.ema.decay, run.owned_names())
    return torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)["update"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--init-from")
    args = parser.parse_args()
    cfg = load_validation_prompts(json.loads(Path(args.config).read_text()))
    validate_config(cfg)

    # PCIe boxes with GPUs on two sockets: NCCL's default stages cross-socket traffic through host shared
    # memory at ~3.4 GB/s; P2P across the socket link measured ~18 GB/s (moss/tools/probe_comms.py).
    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
    # Bucketed aspect ratios vary the activation sizes every micro-step; at ~85 of 96 GB this avoids fragmentation.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(hours=2))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("data",))
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    torch.cuda.manual_seed_all(cfg["seed"] + rank)

    run = FineTune(cfg, rank, world, device, mesh, cpu_group)
    parameters = [p for p in run.model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"[sft] world {world}, {cfg.get('parallelism', 'fsdp')}; trainable "
              f"{sum(p.numel() for p in parameters):,} params (fp32 master); batched forward {run.module.batched}, "
              f"batched text {run.encoder.batched}, checkpointed blocks "
              f"{sum(layer.checkpointed for layer in run.module.layers)}, compile {cfg.get('compile', False)}; "
              f"peak after setup {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    fused = run.replicated
    optimizer = build_optimizers(run.module, cfg, run.owned_names())
    if rank == 0:
        print("[sft] optimizers" + (" (rank 0's owned share)" if run.zero else "") + ": " + "; ".join(
            f"{type(o).__name__} {sum(p.numel() for g in o.param_groups for p in g['params']):,} params"
            for o in optimizer), flush=True)
        if run.zero:
            print(f"[sft] zero: owned elements per rank {run.zero.loads()}, clip "
                  f"{cfg.get('zero', {}).get('clip', 'global')}, {len(run.zero.buckets)} buckets", flush=True)
    # The report's decay-to-one on the last block's q/k norm gains (weight decay toward 1, not 0).
    qk_final = [run.module.layers[-1].q_norm.weight, run.module.layers[-1].k_norm.weight]
    stream = BatchStream(cfg, rank, world)
    output = Path(cfg["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"[sft] {len(stream):,} batches of {world} x {cfg['per_rank_batch']} per epoch; "
              f"{len(stream) // cfg['grad_accum']:,} updates per epoch", flush=True)
    log = JsonLog(output / "metrics.jsonl") if rank == 0 else (lambda **_: None)
    update, previous = 0, 0.0
    if args.resume:
        update, previous = load(args.resume, run, optimizer, stream, cfg)
    elif args.init_from:
        source_update = init_from(args.init_from, run, optimizer)
        if rank == 0:
            print(f"[sft] initialized from {args.init_from} (update {source_update}): model + optimizer", flush=True)
        log(event="init_from", path=str(args.init_from), source_update=source_update)
    for o in optimizer:
        if isinstance(o, torch.optim.AdamW):
            for group in o.param_groups:
                group["fused"], group["foreach"] = (True, False) if fused else (None, None)
    started = time.monotonic()
    elapsed = lambda: previous + time.monotonic() - started
    log(event="start", update=update, resumed=bool(args.resume), world=world)
    if not args.resume and cfg.get("eval_at_start", True):
        run.evaluate(update, log)
    dist.barrier(group=cpu_group)

    stop_requested = {"signal": False}

    def request_stop(*_):
        stop_requested["signal"] = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    stop_file = output / "STOP"
    saver = Saver(run, cfg, cpu_group)
    last_saved = None
    accum = cfg["grad_accum"]
    while update < cfg["max_updates"]:
        flag = torch.tensor(float(stop_requested["signal"] or stop_file.exists()
                                  or elapsed() >= cfg.get("max_seconds", float("inf"))), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            break
        step_started = time.monotonic()
        lr = cfg["lr"] * min(1.0, (update + 1) / cfg["warmup"]) if cfg.get("warmup") else cfg["lr"]
        for o in optimizer:
            o.zero_grad(set_to_none=True)
            for group in o.param_groups:
                group["lr"] = lr
        losses, waits, ts, sizes = [], 0.0, [], set()
        micro_batches = [stream.next() for _ in range(accum)] if cfg.get("encode_update_text") else None
        if micro_batches:
            # One batched encoder pass over the whole update's captions: the encoder is launch-bound at B=2.
            flat = run.encoder([c for _, captions, _ in micro_batches for c in captions])
            per = cfg["per_rank_batch"]
            update_contexts = [flat[i * per:(i + 1) * per] for i in range(accum)]
        for micro in range(accum):
            pixels, captions, wait = micro_batches[micro] if micro_batches else stream.next()
            waits += wait
            sizes.add(f"{pixels.shape[-1]}x{pixels.shape[-2]}")
            with run.sync_context(micro == accum - 1):
                loss, t = run.loss(pixels, captions, update_contexts[micro] if micro_batches else None)
                finite(loss, "loss")
                (loss / accum).backward()
            losses.append(loss.item())
            ts.append(t.mean().item())
        phases, mark = {}, [step_started]

        def lap(name):
            if run.zero is not None:
                torch.cuda.synchronize()
                now = time.monotonic()
                phases[name] = round(now - mark[0], 3)
                mark[0] = now

        lap("backward")
        if run.zero is not None:
            run.zero.reduce_grads()
            lap("reduce")
            norm = run.zero.clip(cfg["max_grad_norm"], cfg.get("zero", {}).get("clip", "global"))
            lap("clip")
        else:
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["max_grad_norm"])
            norm = norm.full_tensor() if hasattr(norm, "full_tensor") else norm
        finite(norm, "grad norm")
        for o in optimizer:
            o.step()
        if cfg.get("qk_norm_decay_to_one"):
            with torch.no_grad():
                for p in qk_final:
                    p.sub_(p - 1, alpha=lr * cfg["qk_norm_decay_to_one"])
        if run.ema is not None:
            run.ema.update(run.module)
        lap("step")
        if run.zero is not None:
            run.zero.broadcast_params()
            lap("broadcast")
        update += 1
        reduced = torch.tensor([sum(losses) / accum, sum(ts) / accum, waits], device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
        seconds = time.monotonic() - step_started
        saver.poll()
        log(event="update", update=update, loss=reduced[0].item(), t_mean=reduced[1].item(), grad_norm=norm.item(),
            lr=lr, seconds=seconds, elapsed=elapsed(), stream_wait=reduced[2].item(), sizes=sorted(sizes),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(), stream=stream.state_dict(),
            **({"phases": phases} if phases else {}))
        if rank == 0:
            print(f"[update {update}/{cfg['max_updates']}] loss={reduced[0].item():.4f} grad={norm.item():.3g} "
                  f"lr={lr:.2e} {','.join(sorted(sizes))} wait {reduced[2].item():.1f}s {seconds:.1f}s "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.0f}GiB"
                  + ("".join(f" {k}={v:.1f}s" for k, v in phases.items()) if phases else ""), flush=True)
        if update % cfg["save_interval"] == 0 or update == cfg["max_updates"]:
            saver.start(optimizer, stream, update, elapsed(),
                        export=update % cfg["milestone_interval"] == 0 or update == cfg["max_updates"])
            last_saved = update
            log(event="checkpoint", update=update)
        if update % cfg["eval_interval"] == 0:
            run.evaluate(update, log)
    if last_saved != update:
        saver.start(optimizer, stream, update, elapsed(), export=True)
        log(event="checkpoint", update=update, final=True)
    saver.finish()
    if stop_file.exists() and rank == 0:
        stop_file.unlink()
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

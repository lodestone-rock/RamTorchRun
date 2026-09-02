"""Shared K2 helpers used by krea2/train.py and krea2/inference.py."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from krea2.model.autoencoder import QwenAutoencoder


# ---------------------------------------------------------------------------
# VAE encode/decode helpers (no gradient, bfloat16)
# ---------------------------------------------------------------------------

@torch.no_grad()
def vae_encode(ae: QwenAutoencoder, pixels: torch.Tensor) -> torch.Tensor:
    """Encode a batch of pixel images [-1, 1] to latent space.

    QwenAutoencoder only exposes a decode() method publicly; we call
    the underlying diffusers AutoencoderKLQwenImage for encoding.
    pixels: [B, 3, H, W] float in [-1, 1].
    Returns: [B, 16, H/8, W/8] bfloat16 latent.
    """
    with torch.autocast("cuda", torch.bfloat16):
        x = pixels.to(ae.ae.device, non_blocking=True)
        x5d = x.unsqueeze(2)  # [B, C, 1, H, W] for qwen-image AE
        enc = ae.ae.encode(x5d)
        latent = enc.latent_dist.mode()          # [B, 16, 1, H/8, W/8]
        latent = latent.squeeze(2)               # [B, 16, H/8, W/8]
        # Normalize with the registered mean/std buffers (same as decode path).
        mean = ae.latents_mean.squeeze(2)        # [1, 16, 1, 1]
        std  = ae.latents_std.squeeze(2)         # [1, 16, 1, 1]
        latent = (latent - mean) / std
        return latent.to(torch.bfloat16)


@torch.no_grad()
def vae_decode(ae: QwenAutoencoder, latent: torch.Tensor) -> torch.Tensor:
    """Decode a latent [B, 16, H/8, W/8] to pixels [B, 3, H, W] in [-1, 1].

    Casts the input to match the AE's weight dtype (bfloat16) so there
    is no float32/bfloat16 mismatch inside the conv layers.
    """
    ae_dtype = next(ae.ae.parameters()).dtype
    with torch.autocast("cuda", ae_dtype):
        return ae.decode(latent.to(ae_dtype))


# ---------------------------------------------------------------------------
# Timestep sampling — K2 resolution-aware shifted schedule (training version)
# ---------------------------------------------------------------------------

def _mu_from_seq_len(
    seq_len: int,
    x1: int,
    x2: int,
    y1: float = 0.5,
    y2: float = 1.15,
) -> float:
    """Linearly interpolate mu from image-token sequence length.

    Mirrors the inference formula in krea2/model/sampling.py::timesteps():
        slope = (y2 - y1) / (x2 - x1)
        mu    = slope * seq_len + (y1 - slope * x1)

    mu = ln(alpha) where alpha is BFL's shift parameter. K2's pretraining used
    a resolution-aware schedule with mu_y1=0.5 increasing to mu_y2=1.15. For
    fine-tuning the pretrained K2 weights keep y1/y2 at the pretrained values
    so timesteps stay in-distribution; for training from scratch on the Qwen
    VAE set mu_override ~ 1.53.
    """
    slope = (y2 - y1) / (x2 - x1)
    return slope * seq_len + (y1 - slope * x1)


def sample_timesteps(
    B: int,
    device: torch.device,
    mu: float,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Sample B timesteps using K2's shifted logit-uniform schedule.

    Exact inverse-CDF of the inference timestep schedule, applied sample-wise:

        u ~ Uniform(0, 1)
        t = exp(mu) / (exp(mu) + (1/u - 1)^sigma)

    With sigma=1 and mu=0 this reduces to Uniform(0,1). mu>0 shifts mass
    toward t=1 (noisier timesteps), matching the shifted schedule used at
    inference for the same resolution.
    """
    # Clamp u away from 0/1 to avoid (1/u - 1) -> inf.
    u = torch.rand(B, device=device).clamp(1e-5, 1 - 1e-5)
    exp_mu = math.exp(mu)
    t = exp_mu / (exp_mu + (1.0 / u - 1.0) ** sigma)
    return t.float()


# ---------------------------------------------------------------------------
# Tag embedding: one policy shared by train.py and train_mass_lora.py
# ---------------------------------------------------------------------------

class TagTrainer:
    """Everything the trainers need to know about the tag block.

    Holds the config, the per-step dropout policy and the `RowAdamW` over the
    table. ``enabled`` is False when no ``tag_embed`` block is configured, and
    every method is then a no-op returning ``(None, None)``, so the call sites
    need no branching.
    """

    def __init__(self, cfg: dict, parallelism: str = "pipeline"):
        tag_cfg = cfg.get("tag_embed") or {}
        self.enabled = bool(tag_cfg.get("enabled", bool(tag_cfg)))
        self.opt = None
        self.dense_opt = None
        self.dense_params: list = []
        if not self.enabled:
            self.vocab_size, self.max_tags = 0, 0
            return

        from utils.tag_vocab import TagVocab

        self.vocab_path = tag_cfg["vocab_path"]
        self.vocab = TagVocab.load(self.vocab_path)
        self.vocab_size = len(self.vocab)
        self.vocab_name = self.vocab.name
        self.tag_dim = tag_cfg.get("tag_dim", 512)
        # The DiT pads its combined sequence to a multiple of 256, so any
        # max_tags in 1..256 costs exactly the same attention. 128 is free.
        self.max_tags = tag_cfg.get("max_tags", 128)
        self.tag_drop_prob = tag_cfg.get("tag_drop_prob", 0.1)
        self.lr = tag_cfg.get("lr", cfg.get("lr", 1e-4))
        self.weight_decay = tag_cfg.get("weight_decay", 0.0)
        self.state_device = tag_cfg.get("state_device", "cpu")
        self.max_grad_norm = tag_cfg.get(
            "max_grad_norm", cfg.get("max_grad_norm", 1.0)
        )

        if "offload" in parallelism:
            print(
                f"  [warn] parallelism={parallelism!r} streams every chunk "
                f"weight per microbatch, and the tag table lives in the embed "
                f"chunk — that is an extra "
                f"{self.vocab_size * self.tag_dim * 2 / 2**30:.2f} GB of "
                f"host-to-device traffic per microbatch. Prefer 'pipeline'."
            )

    def attach(self, dit):
        """Give *dit* its tag table, in place.

        Deliberately NOT done by building the DiT with ``tag_vocab`` set: the
        pretrained checkpoint has no tag keys, so a strict load would fail, and
        under ``assign=True`` from meta the un-loaded table would stay on the
        meta device. Attaching afterwards also satisfies the ordering the LoRA
        paths need — call this after ``inject_lora``/``inject_lora_bank`` (so
        the projection is not adapted) and before any adapter checkpoint load
        (so its ``tagembed.*`` keys land).
        """
        if not self.enabled:
            return dit

        import dataclasses

        from krea2.model.tag_embed import TagEmbedder

        dit.tagembed = TagEmbedder(
            self.vocab_size, dit.config.features, self.tag_dim,
            vocab_name=self.vocab_name,
        ).to(next(dit.parameters()).dtype)
        dit.config = dataclasses.replace(
            dit.config, tag_vocab=self.vocab_size, tag_dim=self.tag_dim
        )
        return dit

    @staticmethod
    def split_checkpoint(sd: dict) -> tuple[dict, dict]:
        """-> (weights without the tag table, the tag table's own state dict).

        The table is attached after the base load, so its keys have to come out
        of a strict load and go back in afterwards via `load_tag_state`.
        """
        pre = "tagembed."
        return (
            {k: v for k, v in sd.items() if not k.startswith(pre)},
            {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)},
        )

    def load_tag_state(self, dit, tag_sd: dict):
        if not tag_sd:
            return
        if not self.enabled or dit.tagembed is None:
            print(f"  [warn] checkpoint carries a tag table ({len(tag_sd)} "
                  f"tensors) but tag_embed is not configured — ignoring it.")
            return
        dit.tagembed.load_state_dict(tag_sd)
        print(f"  Loaded tag table from checkpoint ({len(tag_sd)} tensors).")

    def build_optimizer(self, dit, warmup: int = 0, own_dense: bool = False):
        """Split the tag table out of *dit*'s trainable set into a RowAdamW.

        Returns the parameters the MAIN optimizer should own. The table needs
        its own optimizer because a step touches a few hundred of ~316k rows:
        plain AdamW would decay and drift the other 99.9% every step.

        ``own_dense=True`` also puts the projection and norm in a private
        AdamW. The mass-LoRA trainer needs that: its optimizer is slot-shaped
        and cannot hold a shared dense tensor, and it "freezes" by exclusion
        rather than by ``requires_grad``, so there is no trainable set to
        return into.
        """
        trainable = [p for p in dit.parameters() if p.requires_grad]
        if not self.enabled or dit.tagembed is None:
            return trainable

        from torch.optim import AdamW

        from krea2.model.tag_embed import check_vocab, tag_embed_bytes_report
        from utils.row_optimizer import RowAdamW

        # If a checkpoint was loaded it overwrote the fingerprint buffer, so
        # this compares the TRAINED id space against the configured one.
        check_vocab(dit.tagembed, self.vocab_size, self.vocab_name)

        table = dit.tagembed.embed.weight
        self.opt = RowAdamW(
            [table], self.vocab_size, lr=self.lr, betas=(0.9, 0.95),
            weight_decay=self.weight_decay, warmup=warmup,
            state_device=self.state_device,
        )
        self.dense_params = [
            p for p in dit.tagembed.parameters() if p is not table
        ]
        print("  Tag embedding: " + tag_embed_bytes_report(
            self.vocab_size, self.tag_dim, dit.tagembed.proj.out_features,
            table.dtype, torch.float32,
            state_on_host=(self.state_device == "cpu"),
        ))
        print(f"    vocab={self.vocab_path} max_tags={self.max_tags} "
              f"lr={self.lr:g} tag_drop_prob={self.tag_drop_prob:g}")

        if own_dense:
            self.dense_opt = AdamW(
                self.dense_params, lr=self.lr, betas=(0.9, 0.95),
                weight_decay=self.weight_decay,
            )
            # By id(): `p in list_of_tensors` would run elementwise __eq__ and
            # raise on the first shape mismatch.
            owned = {id(table)} | {id(p) for p in self.dense_params}
            return [p for p in trainable if id(p) not in owned]
        # The projection and norm are dense — every step touches all of them —
        # so they belong in the caller's main optimizer.
        return [p for p in trainable if p is not table]

    def batch(self, batch_data, uncond_flags, device):
        """-> (tag_ids, tag_mask) for this batch, or (None, None).

        *uncond_flags* is the per-sample list the caller already computed for
        caption blanking. Unconditional samples lose their tags TOO — a CFG
        negative pass drops both channels, so training has to match. A separate
        ``tag_drop_prob`` coin drops the tags alone, which is what teaches the
        model that a caption without tags is still a valid prompt.
        """
        if not self.enabled:
            return None, None
        # Two batch shapes: parquet_dataloader appends a trailing extras dict
        # to its tuple, mass_lora_dataloader merges the same keys into the dict
        # it already returns.
        extras = batch_data if isinstance(batch_data, dict) else (
            batch_data[-1] if isinstance(batch_data[-1], dict) else None
        )
        if extras is None or "tag_ids" not in extras:
            raise RuntimeError(
                "tag_embed is configured but the dataloader returned no tag "
                "block — set parquet_dataloader.tag_column."
            )
        tag_ids = extras["tag_ids"].to(device, non_blocking=True)
        tag_mask = extras["tag_mask"].to(device, non_blocking=True)
        # Previews render the ORIGINAL captions, so they want the tags this row
        # actually has, before dropout.
        self.last_undropped = (tag_ids, tag_mask)

        drop = torch.tensor(
            [u or (torch.rand(1).item() < self.tag_drop_prob)
             for u in uncond_flags],
            device=device, dtype=torch.bool,
        )
        tag_mask = tag_mask & ~drop[:, None]
        return tag_ids, tag_mask

    def undropped(self):
        """The batch's tags with no dropout applied, for previews."""
        return getattr(self, "last_undropped", (None, None))

    @staticmethod
    def active_rows(tag_ids, tag_mask):
        """The vocabulary rows this step actually used."""
        return tag_ids[tag_mask] if tag_ids is not None else None

    def step(self, tag_ids, tag_mask):
        """Update the table's active rows. Call next to the main opt.step()."""
        if self.opt is None or tag_ids is None:
            return
        from utils.row_optimizer import clip_row_grads

        rows = self.active_rows(tag_ids, tag_mask)
        if rows.numel():
            # Clipped separately from the main model: a global norm over 162M
            # mostly-zero rows would be dominated by the table's size rather
            # than by what this step actually learned.
            clip_row_grads(self.opt.params, self.max_grad_norm, rows)
            self.opt.step(rows)
        self.opt.zero_grad()

        if self.dense_opt is not None:
            torch.nn.utils.clip_grad_norm_(self.dense_params, self.max_grad_norm)
            self.dense_opt.step()
            self.dense_opt.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# SDPA backend pinning
# ---------------------------------------------------------------------------

_SDPA_PIN_CTX = None


def _pin_sdpa_backends():
    """Pin SDPA backend selection once, process-globally.

    krea2.model.mmdit.attention wraps SDPA in ``sdpa_kernel(CUDNN)``, which
    mutates process-global flags on enter/exit — racing across the pipeline's
    stage worker threads and eventually leaving the process with every backend
    disabled ("No available kernel", first seen in the VAE decode). Instead:
    disable that per-call context and pin the backend PRIORITY once. Order
    matters: the torch default is [flash, mem_efficient, MATH, cudnn] — math
    outranks cudnn, so for the DiT's bool-masked attention (flash/mem-eff
    decline it) math would materialize the O(heads * L^2) fp32 score matrix
    instead of using cudnn's fused kernel. cudnn first; math stays last as the
    only backend that serves the VAE's head_dim-512 attention.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import krea2.model.mmdit

    # The context must stay referenced for the process lifetime: if the
    # generator-based context manager is garbage-collected, GeneratorExit runs
    # its finally block and silently RESTORES the default priority.
    global _SDPA_PIN_CTX
    _SDPA_PIN_CTX = sdpa_kernel(
        [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
         SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
        set_priority=True,
    )
    _SDPA_PIN_CTX.__enter__()  # never exited: pinned for the whole process
    krea2.model.mmdit.set_sdpa_ctx(False)

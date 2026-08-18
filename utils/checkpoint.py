"""Checkpoint helpers shared by the trainer and inference scripts.

Extracted from x0-pred's krea2_trainer.py and expand_lora_rank.py:
  - _strip_compiled_keys:    drop torch.compile's `_orig_mod.` prefixes
  - load_lora_checkpoint:    partial (strict=False) LoRA state-dict load
  - merge_lora_into_base_sd: bake a LoRA checkpoint into a full base
    state dict (full-FT resume from a LoRA run)
  - _classify_keys / _infer_old_rank: LoRA key-convention detection
"""
from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import load_file


def _strip_compiled_keys(sd: dict) -> dict:
    prefix = "_orig_mod."
    return {k.replace(prefix, "") if prefix in k else k: v for k, v in sd.items()}


def load_lora_checkpoint(model: nn.Module, path: str):
    """Load a LoRA checkpoint (partial state dict — missing keys are fine)."""
    if path.endswith((".safetensors", ".sft")):
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[ckpt] {len(unexpected)} unexpected keys: {unexpected[:5]} ...")
    print(f"[ckpt] Loaded LoRA ckpt from {path} "
          f"({len(sd)} tensors, {len(missing)} missing keys).")


def _classify_keys(sd: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str], str]:
    """Split state-dict keys into (down_keys, up_keys, other, convention).

    Auto-detects the LoRA naming convention. In every convention the
    down-projection (A) has shape (rank, in) and the up-projection (B)
    has shape (out, rank), so downstream logic is convention-agnostic.

    Supported conventions (checked in order, first match wins):
      - "k2":        `*.lora_A` / `*.lora_B`            (this repo, bare params)
      - "k2_weight": `*.lora_A.weight` / `*.lora_B.weight`
      - "peft":      `*lora_down.weight` / `*lora_up.weight`   (diffusers/PEFT)
      - "peft_dot":  `*lora.down.weight` / `*lora.up.weight`
      - "peft_bare": `*.lora_down` / `*.lora_up`
    """
    _CONVENTIONS = [
        ("k2",        lambda k: k.endswith(".lora_A"),          lambda k: k.endswith(".lora_B")),
        ("k2_weight", lambda k: k.endswith(".lora_A.weight"),   lambda k: k.endswith(".lora_B.weight")),
        ("peft_dot",  lambda k: "lora.down.weight" in k,        lambda k: "lora.up.weight" in k),
        ("peft",      lambda k: "lora_down.weight" in k,        lambda k: "lora_up.weight" in k),
        ("peft_bare", lambda k: k.endswith(".lora_down"),       lambda k: k.endswith(".lora_up")),
    ]
    for name, is_down, is_up in _CONVENTIONS:
        down = [k for k in sd if is_down(k)]
        up = [k for k in sd if is_up(k)]
        if down and up:
            other = [k for k in sd if k not in set(down) | set(up)]
            return down, up, other, name
    # Nothing matched — report what we found for debugging.
    lora_like = [k for k in sd if "lora" in k.lower()]
    raise ValueError(
        f"No recognised LoRA key convention found. "
        f"LoRA-ish keys present (first 10): {lora_like[:10] or 'NONE'}"
    )


def _infer_old_rank(sd: dict[str, torch.Tensor], down_keys: list[str]) -> int:
    """Infer the rank from the first down-proj tensor's dim 0; validate all agree."""
    if not down_keys:
        raise ValueError("No down-projection keys found — not a LoRA checkpoint?")
    ranks = {sd[k].shape[0] for k in down_keys}
    if len(ranks) != 1:
        raise ValueError(
            f"Inconsistent ranks across down-proj tensors: {sorted(ranks)}. "
            "All down-proj tensors must share dim 0 (the rank)."
        )
    return ranks.pop()


def merge_lora_into_base_sd(
    base_sd: dict,
    lora_sd: dict,
    rank: int,
    alpha: float,
    device: str = "cpu",
) -> dict:
    """Merge a K2 LoRA checkpoint into a full base DiT state dict (in-place).

        W_new = W_base + (alpha / rank) * (lora_B @ lora_A)   (fp32 matmul)

    Non-LoRA trainables (RMSNorm scales, modulation tensors, biases) override
    the base values byte-for-byte — they were trained alongside the adapters.

    Returns the modified base_sd.
    """
    scale = alpha / rank
    a_keys, b_keys, other_keys, _conv = _classify_keys(lora_sd)
    ckpt_rank = _infer_old_rank(lora_sd, a_keys)
    if ckpt_rank != rank:
        raise ValueError(
            f"[merge] config lora_rank={rank} but checkpoint rank={ckpt_rank}. "
            f"Set lora_rank to match the LoRA checkpoint."
        )

    merged = 0
    missing_base = []
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A", ".lora_B")
        if b_key not in lora_sd:
            print(f"[merge] [warn] no matching lora_B for {a_key}; skipping")
            continue
        base_key = a_key[: -len(".lora_A")] + ".weight"
        if base_key not in base_sd:
            missing_base.append(base_key)
            continue

        A = lora_sd[a_key].to(device, torch.float32)      # (rank, in)
        B = lora_sd[b_key].to(device, torch.float32)      # (out, rank)
        W = base_sd[base_key].to(device, torch.float32)   # (out, in)
        W_new = W + scale * (B @ A)
        base_sd[base_key] = W_new.to(base_sd[base_key].dtype).cpu()
        merged += 1

    overridden = 0
    for k in other_keys:
        if k in base_sd and base_sd[k].shape != lora_sd[k].shape:
            print(f"[merge] [warn] shape mismatch on override {k}: "
                  f"base {base_sd[k].shape} vs lora {lora_sd[k].shape}; skipping")
            continue
        base_sd[k] = lora_sd[k].clone()
        overridden += 1

    if missing_base:
        print(f"[merge] [warn] {len(missing_base)} LoRA targets had no matching "
              f"base weight (first 5): {missing_base[:5]}")
    print(f"[merge] Merged {merged} LoRA deltas, overrode {overridden} "
          f"non-LoRA trainables (scale={scale:.4f}).")
    return base_sd

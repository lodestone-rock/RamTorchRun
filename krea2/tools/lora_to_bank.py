"""lora_to_bank.py — wrap a single LoRA file as a 1-slot mass-LoRA bank.

The inverse of ``export_lora_bank.slice_slot``: ``*.lora_A`` [r, in] becomes
``*.lora_A_bank`` [1, r, in] (same for B), so ``train_mass_lora.py``'s
``bank_checkpoint`` evaluation (held-out loss) scores adapters trained by
``train_concept_lora.py``. Any non-adapter trainable key (norms, modulation,
concept embedder) has no bank equivalent: refused unless --drop-extra.

    uv run python krea2/tools/lora_to_bank.py concept_step_4000_final.safetensors -o bank.safetensors
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors.torch import load_file, save_file

from krea2.tools.export_lora_bank import slice_slot


def to_bank(sd: dict, dtype=torch.float32, drop_extra: bool = False) -> dict:
    out, extra = {}, []
    for key, tensor in sd.items():
        if key.endswith((".lora_A", ".lora_B")):
            out[key + "_bank"] = tensor.to(dtype).unsqueeze(0).contiguous()
        else:
            extra.append(key)
    if extra and not drop_extra:
        raise ValueError(f"{len(extra)} non-adapter keys have no bank form "
                         f"(e.g. {extra[:3]}); pass --drop-extra to discard them")
    if not out:
        raise ValueError("no '*.lora_A' / '*.lora_B' tensors found")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lora")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--drop-extra", action="store_true")
    args = ap.parse_args()
    sd = load_file(args.lora)
    bank = to_bank(sd, drop_extra=args.drop_extra)
    back = slice_slot(bank, 0)
    for key, tensor in back.items():
        if not torch.equal(tensor, sd[key].float()):
            raise AssertionError(f"round trip mismatch at {key}")
    save_file(bank, args.out)
    print(f"{len(bank)} bank tensors -> {args.out} (slice_slot round trip exact)")


if __name__ == "__main__":
    main()

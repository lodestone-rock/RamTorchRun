"""export_lora_bank.py — Slice a mass-LoRA bank into standard LoRA files.

`krea2/train_mass_lora.py` saves ONE checkpoint holding every slot
(``*.lora_A_bank`` of shape ``[L, rank, in]``). Everything else in the repo —
`krea2/inference.py --lora-checkpoint`, `utils/checkpoint.py`'s merge helpers,
`krea2/train.py`'s ``lora_checkpoint`` resume — speaks the single-adapter
convention (``*.lora_A`` / ``*.lora_B``). This tool converts.

Slots are named from the ``slots.json`` the trainer writes next to the bank, so
the output files carry the group value (artist / character / ...) rather than an
index.

`inject_lora_bank` skips layers where the adapter would be no smaller than the
weight it adapts (K2's ``txtfusion.projector``, a ``Linear(12, 1)``), so
`utils/checkpoint.load_lora_checkpoint` reports those two keys as missing when
an exported file is loaded into a fully `inject_lora`'d model. That is benign:
an un-loaded adapter keeps ``lora_B = 0`` and contributes exactly nothing.

Run:
    # every slot -> runs/<run>/ckpts/exported/lora_<name>.safetensors
    uv run python krea2/tools/export_lora_bank.py runs/k2-mass/ckpts/bank_step_2000.safetensors

    # one slot, by index or by name, to an explicit path
    uv run python krea2/tools/export_lora_bank.py <bank.safetensors> --slot 7 -o out.safetensors
    uv run python krea2/tools/export_lora_bank.py <bank.safetensors> --name artist_foo
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from safetensors.torch import load_file, save_file


def _slot_name_of(names: list[str] | None, slot: int) -> str:
    if names and slot < len(names):
        return names[slot]
    return f"slot{slot}"


def _safe(name: str, limit: int = 64) -> str:
    return "".join(
        ch if ch.isalnum() or ch in "-_." else "_" for ch in name
    )[:limit]


def load_bank(path: str):
    """(bank state dict, slots.json metadata or None, n_slots)."""
    sd = load_file(path, device="cpu")
    a_keys = [k for k in sd if k.endswith(".lora_A_bank")]
    if not a_keys:
        raise ValueError(
            f"{path} has no '*.lora_A_bank' tensors — is it a mass-LoRA bank? "
            f"(first keys: {list(sd)[:3]})"
        )
    n_slots = {sd[k].shape[0] for k in a_keys}
    if len(n_slots) != 1:
        raise ValueError(f"inconsistent slot counts across banks: {sorted(n_slots)}")

    meta = None
    meta_path = os.path.join(os.path.dirname(os.path.abspath(path)), "slots.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return sd, meta, n_slots.pop()


def slice_slot(sd: dict, slot: int) -> dict:
    """One slot under the standard ``.lora_A`` / ``.lora_B`` convention."""
    out = {}
    for key, tensor in sd.items():
        if key.endswith(".lora_A_bank"):
            out[key[: -len(".lora_A_bank")] + ".lora_A"] = (
                tensor[slot].clone().contiguous()
            )
        elif key.endswith(".lora_B_bank"):
            out[key[: -len(".lora_B_bank")] + ".lora_B"] = (
                tensor[slot].clone().contiguous()
            )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bank", help="path to bank_step_N.safetensors")
    ap.add_argument("--slot", type=int, help="export only this slot index")
    ap.add_argument("--name", help="export only the slot with this group name")
    ap.add_argument("-o", "--out", help="output file (single slot) or directory")
    ap.add_argument("--min-steps", type=int, default=0,
                    help="skip slots that trained fewer than this many steps "
                         "(needs slots.json)")
    args = ap.parse_args()

    sd, meta, n_slots = load_bank(args.bank)
    names = meta.get("slot_names") if meta else None
    steps = meta.get("slot_steps") if meta else None
    rank = next(iter(sd.values())).shape[1]
    print(f"Bank: {len(sd)} tensors, {n_slots} slots, rank {rank}"
          + (", named from slots.json" if names else ", no slots.json found"))

    if args.name is not None:
        if not names:
            raise SystemExit("--name needs a slots.json next to the bank")
        if args.name not in names:
            raise SystemExit(
                f"{args.name!r} is not a slot; first few: {names[:5]}"
            )
        targets = [names.index(args.name)]
    elif args.slot is not None:
        if not 0 <= args.slot < n_slots:
            raise SystemExit(f"--slot must be in [0, {n_slots})")
        targets = [args.slot]
    else:
        targets = list(range(n_slots))

    if args.min_steps > 0:
        if not steps:
            raise SystemExit("--min-steps needs a slots.json next to the bank")
        kept = [s for s in targets if steps[s] >= args.min_steps]
        print(f"  {len(targets) - len(kept)} slot(s) below --min-steps "
              f"{args.min_steps}, skipping.")
        targets = kept

    single = len(targets) == 1 and args.out and not args.out.endswith(os.sep)
    if single and os.path.isdir(args.out):
        single = False
    if single:
        out_paths = {targets[0]: args.out}
    else:
        out_dir = args.out or os.path.join(
            os.path.dirname(os.path.abspath(args.bank)), "exported"
        )
        os.makedirs(out_dir, exist_ok=True)
        out_paths = {
            s: os.path.join(
                out_dir, f"lora_{_safe(_slot_name_of(names, s))}.safetensors"
            )
            for s in targets
        }

    for slot in targets:
        slot_sd = slice_slot(sd, slot)
        save_file(slot_sd, out_paths[slot])
        step_note = f", {steps[slot]} steps" if steps else ""
        print(f"  slot {slot} ({_slot_name_of(names, slot)}{step_note}) -> "
              f"{out_paths[slot]} [{len(slot_sd)} tensors]")
    print(f"Exported {len(targets)} adapter(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

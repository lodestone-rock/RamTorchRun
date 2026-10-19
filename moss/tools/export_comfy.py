"""Release-layout ``transformer/`` folder -> one ComfyUI ``diffusion_models`` file (Comfy-Org/MOSS-Image layout).

The only difference is that each block's ``proj.{q,k,v}`` is fused into ``proj.qkv`` (q, k, v stacked on the
output dim); dtypes are kept (bf16, ``rope.bands`` fp32). ComfyUI maps LoRAs on the separate q/k/v onto the
fused slices, so kohya LoRAs trained on the release layout load unchanged.

uv run python moss/tools/export_comfy.py runs/moss-base-sft-v2/checkpoints/update-002000/transformer \
    kromoss_bf16.safetensors [--compare COMFY_ORG_FILE]
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(folder):
    weight_map = json.loads((folder / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(folder / shard), "pt") as f:
            state.update({k: f.get_tensor(k) for k in f.keys()})
    for q in [k for k in state if k.endswith(".proj.q.weight")]:
        prefix = q[:-len("q.weight")]
        state[f"{prefix}qkv.weight"] = torch.cat([state.pop(f"{prefix}{p}.weight") for p in "qkv"])
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transformer")
    parser.add_argument("output")
    parser.add_argument("--compare", help="a Comfy-Org file: require identical keys, shapes and dtypes; report "
                                          "how many tensors are bit-identical")
    args = parser.parse_args()
    state = convert(Path(args.transformer))
    if args.compare:
        with safe_open(args.compare, "pt") as f:
            if set(f.keys()) != set(state):
                raise ValueError(f"key sets differ: {sorted(set(f.keys()) ^ set(state))[:5]}")
            equal = 0
            for k in f.keys():
                ref = f.get_tensor(k)
                if ref.shape != state[k].shape or ref.dtype != state[k].dtype:
                    raise ValueError(f"{k}: {tuple(ref.shape)} {ref.dtype} vs {tuple(state[k].shape)} {state[k].dtype}")
                equal += torch.equal(ref, state[k])
        print(f"layout matches {args.compare}; {equal}/{len(state)} tensors bit-identical")
    save_file(state, args.output)
    print(f"{args.output}: {len(state)} tensors")


if __name__ == "__main__":
    main()

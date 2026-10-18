"""Split a data-space DMD ``generator.safetensors`` into standalone adapter files (fp32):

    lora.safetensors            kohya-style: diffusion_model.<module>.lora_down.weight (A) / lora_up.weight (B) / alpha
    time_embedding.safetensors  stock-key overlay for the bf16 base: time_embedding.* (the step embedder) plus
                                shared_block_modulation.bias / video_head_modulation.bias, which carry the folded
                                time(1) vector and are REQUIRED with it (the embedder alone is wrong)

Q and the step timesteps stay in dataspace.safetensors.

uv run python moss/tools/export_lora.py GENERATOR.safetensors OUT_DIR [--alpha 128]
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from pathlib import Path

import torch

TIME_KEYS = ("time_embedding.", "shared_block_modulation.bias", "video_head_modulation.bias")


def adapter_files(export: dict[str, torch.Tensor], alpha: float, out: Path):
    from safetensors.torch import save_file
    lora = {}
    for key in sorted(export):
        if key.endswith(".lora_A"):
            module = key[:-len(".lora_A")]
            lora[f"diffusion_model.{module}.lora_down.weight"] = export[key].float().contiguous()
            lora[f"diffusion_model.{module}.lora_up.weight"] = export[f"{module}.lora_B"].float().contiguous()
            lora[f"diffusion_model.{module}.alpha"] = torch.tensor(float(alpha))
    rank = next(v for k, v in lora.items() if k.endswith("lora_down.weight")).shape[0]
    save_file(lora, str(out / "lora.safetensors"), metadata={"rank": str(rank), "alpha": str(alpha)})
    time = {k: v.float().contiguous() for k, v in export.items() if k.startswith(TIME_KEYS)}
    save_file(time, str(out / "time_embedding.safetensors"))
    return len(lora) // 3, sorted(time)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generator")
    parser.add_argument("out")
    parser.add_argument("--alpha", type=float, default=128.0)
    args = parser.parse_args()
    from safetensors.torch import load_file
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modules, time_keys = adapter_files(load_file(args.generator), args.alpha, out)
    print(f"{out}: lora.safetensors ({modules} modules), time_embedding.safetensors {time_keys}")


if __name__ == "__main__":
    main()

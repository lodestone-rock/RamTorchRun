"""Export a data-space DMD checkpoint as a release-format MOSS ``transformer/`` folder (merged EMA weights).

Reads ``trainer.pt`` (EMA generator), so any saved checkpoint works, not only milestones. Writes

    OUT/transformer/config.json + diffusion_pytorch_model-*.safetensors + index   (stock MOSSImageTransformer)
    OUT/dataspace.safetensors   latent_mix.Q [128, 128] and step_codes [R] (fp32)
    OUT/generator.safetensors   the unmerged fp32 export (LoRA + time MLP + folded biases), as the trainer writes
    OUT/lora.safetensors + time_embedding.safetensors   the same split into adapter files (moss/tools/export_lora.py)
    OUT/README.md

The sampler is NOT the release scheduler loop: x0 = 0; for r < R: x0 = z - transformer(z + Q x0, step_codes[r]).
The merged weights default to fp32: a bf16 merge rounds away most of the DMD delta (``--dtype bfloat16 --report``
prints how much; at update 600 the rounding error was ~95% of the delta's norm).

uv run python moss/tools/export_diffusers.py RUN/checkpoints/update-XXXXXX OUT
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import shutil
from pathlib import Path

import torch

from moss.model.transformer import load_transformer
from moss.tools.export_lora import adapter_files
from moss.train_dataspace_dmd import MossDataSpaceDMD

HF_IDS = {"moss-image-sft": "OpenMOSS-Team/MOSS-Image-SFT", "moss-image-base": "OpenMOSS-Team/MOSS-Image-Base"}
DEMOS = {"moss-image-sft": "# Demo\nhttps://huggingface.co/spaces/hugging-apps/moss-image-8-step-dmd\n\n"}

README = """---
license: apache-2.0
base_model: {base_id}
---
{demo}# {base_name}, {steps}-step data-space DMD

Work-in-progress distillation of [{base_id}](https://huggingface.co/{base_id})
(50 steps, CFG 3) to {steps} network evaluations, no CFG. Data-space DMD (arXiv 2610.12102): the
generator iterates on its own clean estimate from fixed noise. Trained on compact-JSON `scene_graph` captions.

Checkpoint: update {update} (EMA), {dtype} weights.

# Usage (recommended): the LoRA alone, standard sampler

Load `lora.safetensors` on the stock {base_name} transformer (e.g. a ComfyUI LoRA loader, strength 1) and
sample with the model's usual flow-matching Euler sampler, shift 5, **CFG 1**, 4-8 steps. Nothing else from
this repo is needed.

# Data-space sampler (the training-time form)

`transformer/` is a stock `MOSSImageTransformer` (load it with the release `pipeline_moss_image.py`); the
text encoder, tokenizer and VAE are the base repo's.

Precision: the distillation delta is smaller than bf16's rounding step, so `transformer/` must be loaded AND run
in fp32 (`torch_dtype=torch.float32`, no autocast); casting it to bf16 discards most of the distillation. For fast
bf16 inference keep the base repo's bf16 transformer and apply the adapter files unmerged:

- `lora.safetensors`: fp32 LoRA, rank 128 / alpha 128 on every block's `proj.{{q,k,v,o,ffn_up_gate,ffn_down}}`,
  kohya keys `diffusion_model.<module>.lora_down.weight` / `lora_up.weight` / `alpha`; apply as
  `y = W x + (alpha / rank) * up(down(x))`.
- `time_embedding.safetensors`: fp32 replacements for `time_embedding.*` (the step embedder) AND
  `shared_block_modulation.bias` / `video_head_modulation.bias`; the t=1 time vector is folded into those biases,
  so the embedder only works together with them. Keep these three modules in fp32.
- `generator.safetensors`: the same tensors in one file (`<module>.lora_A/B` keys) plus Q and the step codes.

The sampler differs from the release scheduler:

```python
from safetensors.torch import load_file
ds = load_file("dataspace.safetensors")            # latent_mix.Q [128, 128], step_codes [{steps}]
Q, codes = ds["latent_mix.Q"].cuda(), ds["step_codes"].cuda()
z = torch.randn(1, 128, H // 16, W // 16, device="cuda")   # fp32
x0 = torch.zeros_like(z)
for r in range({steps}):                            # fewer steps also work (each prefix is trained)
    x = z + torch.einsum("oc,bchw->bohw", Q, x0)
    v = transformer(x, codes[r], prompt_embeds, prompt_mask).sample.float()
    x0 = z - v
# decode x0 with the release VAE (BN denormalize + 2x2 unpatchify)
```
"""


def trainer_export(ckpt, cfg):
    """fold_export of the EMA generator stored in ``trainer.pt``."""
    trainer = torch.load(ckpt / "trainer.pt", map_location="cpu", weights_only=False)
    shadow = trainer["ema"]["shadow"]
    del trainer
    model = load_transformer(Path(cfg["model_path"]) / "transformer", "cpu")
    run = MossDataSpaceDMD(cfg, model, torch.device("cpu"), conditioners=False)
    with torch.no_grad():
        for n, p in run.gen_named.items():
            p.copy_(shadow[n])
    return run.export_state()


def merged_state(base, export, scale, dtype):
    """Stock transformer state dict: base + scale * B A per LoRA module, folded time path; plus the
    bf16 delta-survival report."""
    state, report = {}, []
    for key in export:
        if not key.endswith(".lora_A"):
            continue
        name = key[:-len(".lora_A")]
        weight = base[f"{name}.weight"].float()
        delta = (export[f"{name}.lora_B"].float() @ export[key].float()) * scale
        merged = (weight + delta).to(dtype)
        if dtype != torch.float32:
            kept = (merged.float() - weight).norm() / delta.norm().clamp_min(1e-30)
            lost = (merged.float() - weight - delta).norm() / delta.norm().clamp_min(1e-30)
            report.append((name, kept.item(), lost.item()))
        state[f"{name}.weight"] = merged
    for key, value in base.items():
        state.setdefault(key, value.to(dtype))
    for key, value in export.items():
        if key.startswith(("time_embedding.", "shared_block_modulation.bias", "video_head_modulation.bias")):
            state[key] = value.to(dtype)
    if set(state) != set(base):
        raise ValueError(f"merged keys differ from the base: {sorted(set(state) ^ set(base))[:5]}")
    return state, report


def base_state(folder):
    from safetensors.torch import load_file
    index = json.loads((folder / "diffusion_pytorch_model.safetensors.index.json").read_text())
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update({k: v.to(torch.bfloat16) for k, v in load_file(str(folder / shard)).items()})
    return state


def write_shards(state, folder, max_bytes=5 * 2**30):
    from safetensors.torch import save_file
    folder.mkdir(parents=True, exist_ok=True)
    shards, current, size = [], {}, 0
    for key in sorted(state):
        nbytes = state[key].numel() * state[key].element_size()
        if current and size + nbytes > max_bytes:
            shards.append(current)
            current, size = {}, 0
        current[key] = state[key].contiguous()
        size += nbytes
    shards.append(current)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        name = f"diffusion_pytorch_model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(folder / name), metadata={"format": "pt"})
        for key, value in shard.items():
            weight_map[key] = name
            total += value.numel() * value.element_size()
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    from safetensors.torch import load_file, save_file

    ckpt = Path(args.checkpoint)
    cfg = json.loads((ckpt.parents[1] / "config.json").read_text())
    update = json.loads((ckpt / "metadata.json").read_text())["update"]
    torch.set_num_threads(min(64, os.cpu_count()))
    if (ckpt / "trainer.pt").exists():
        export = trainer_export(ckpt, cfg)
    else:
        export = load_file(str(ckpt / "generator.safetensors"))
    dtype = getattr(torch, args.dtype)
    base_folder = Path(cfg["model_path"]) / "transformer"
    state, report = merged_state(base_state(base_folder), export, cfg["lora_alpha"] / cfg["lora_rank"], dtype)
    steps = len(export["step_codes"])
    if args.report and report:
        kept = sorted(r[1] for r in report)
        lost = sorted(r[2] for r in report)
        print(f"bf16 merge over {len(report)} Linears: |rounded delta|/|delta| median {kept[len(kept) // 2]:.3f} "
              f"(min {kept[0]:.3f}); |rounding error|/|delta| median {lost[len(lost) // 2]:.3f} (max {lost[-1]:.3f})")

    out = Path(args.out)
    if (out / "transformer").exists():
        shutil.rmtree(out / "transformer")
    write_shards(state, out / "transformer")
    shutil.copy(base_folder / "config.json", out / "transformer" / "config.json")
    save_file({"latent_mix.Q": export["latent_mix.Q"].contiguous(), "step_codes": export["step_codes"].contiguous()},
              str(out / "dataspace.safetensors"))
    save_file({k: v.contiguous() for k, v in export.items()}, str(out / "generator.safetensors"))
    adapter_files(export, cfg["lora_alpha"], out)
    key = Path(cfg["model_path"]).name
    base_id = HF_IDS.get(key, cfg["model_path"])
    (out / "README.md").write_text(README.format(steps=steps, update=update, dtype=args.dtype, base_id=base_id,
                                                 base_name=base_id.split("/")[-1], demo=DEMOS.get(key, "")))
    (out / "export.json").write_text(json.dumps({"update": update, "checkpoint": str(ckpt), "dtype": args.dtype}))
    print(f"exported update {update} -> {out}", flush=True)


if __name__ == "__main__":
    main()

# checkpoints/

This folder holds model weights. Everything except this README is gitignored.

## Required assets

| Asset | How to get it | Where the code expects it |
|---|---|---|
| K2 DiT base weights (`SingleStreamDiT`, `large_wide` = ~12B params, safetensors) | Local file — copy or symlink your Krea-2 base checkpoint here | `mmdit_checkpoint` in `configs/train_pipeline_*.json` (default `checkpoints/krea2/raw.safetensors`), or `--mmdit-checkpoint` for `inference.py` |
| Qwen-Image VAE | Auto-downloaded from HuggingFace (`Qwen/Qwen-Image`, subfolder `vae`) on first run | HF cache (`~/.cache/huggingface` or `$HF_HOME`) |
| Qwen3-VL-4B text encoder | Auto-downloaded from HuggingFace (`Qwen/Qwen3-VL-4B-Instruct`) on first run | HF cache |

## Layout convention

```
checkpoints/
├── README.md          # this file (tracked)
└── krea2/
    └── raw.safetensors    # K2 DiT base weights (bring your own)
```

Training runs write their own checkpoints to `runs/<run-name>/ckpts/`
(`lora_step_N.safetensors` or `full_step_N.safetensors`), which `inference.py`
can consume directly via `--lora-checkpoint` / `--mmdit-checkpoint`.

Note: the `mmdit_checkpoint` key in a config accepts any path — absolute paths
to checkpoints elsewhere on disk work fine.

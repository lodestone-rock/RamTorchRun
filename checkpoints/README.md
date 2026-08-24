# checkpoints/

This folder holds model weights. Everything except this README is gitignored.

## Required assets

| Asset | How to get it | Where the code expects it |
|---|---|---|
| K2 DiT base weights (`SingleStreamDiT`, `large_wide` = ~12B params, safetensors) | Local file — copy or symlink your Krea-2 base checkpoint here | `mmdit_checkpoint` in `krea2/configs/*.json` (default `checkpoints/krea2/raw.safetensors`), or `--mmdit-checkpoint` for `krea2/inference.py` |
| Qwen-Image VAE | Auto-downloaded from HuggingFace (`Qwen/Qwen-Image`, subfolder `vae`) on first run | HF cache (`~/.cache/huggingface` or `$HF_HOME`) |
| Qwen3-VL-4B text encoder | Auto-downloaded from HuggingFace (`Qwen/Qwen3-VL-4B-Instruct`) on first run | HF cache |
| Chroma1-HD base weights (~8.9B, 643 tensors, native flow keys) | `hf download lodestones/Chroma1-HD` — or symlink a local copy | `chroma_checkpoint` in `chroma/configs/*.json` (default `checkpoints/chroma/Chroma1-HD.safetensors`) |
| T5-XXL text encoder + tokenizer + flux VAE | Auto-downloaded from HuggingFace (`lodestones/Chroma1-HD`, subfolders `text_encoder` / `tokenizer` / `vae`) on first run | HF cache |

## Layout convention

```
checkpoints/
├── README.md          # this file (tracked)
├── krea2/
│   └── raw.safetensors            # K2 DiT base weights (bring your own)
└── chroma/
    └── Chroma1-HD.safetensors     # Chroma1-HD base weights
```

## Run outputs

Training runs write their own checkpoints to `runs/<run-name>/ckpts/`
(`lora_step_N.safetensors` or `full_step_N.safetensors`; TDM runs write
`tdm_student_step_N.safetensors` plus a resumable
`tdm_state_step_N.safetensors`). The matching `inference.py` consumes them
directly via `--lora-checkpoint`, or via `--mmdit-checkpoint` /
`--chroma-checkpoint` for a merged full model. TDM
student checkpoints use the standard LoRA key convention, so
`--lora-checkpoint ... --steps 4 --guidance 0` runs them as-is.

Note: every `*_checkpoint` key in a config accepts any path — absolute paths to
checkpoints elsewhere on disk work fine.

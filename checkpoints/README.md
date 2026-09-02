# checkpoints/

This folder holds model weights. Everything except this README is gitignored.

## Required assets

| Asset | How to get it | Where the code expects it |
|---|---|---|
| K2 DiT base weights (`SingleStreamDiT`, `large_wide` = ~12B params, safetensors) | Local file — copy or symlink your Krea-2 base checkpoint here | `mmdit_checkpoint` in `krea2/configs/*.json` (default `checkpoints/krea2/raw.safetensors`), or `--mmdit-checkpoint` for `krea2/inference.py` |
| Qwen-Image VAE | Auto-downloaded from HuggingFace (`Qwen/Qwen-Image`, subfolder `vae`) on first run | HF cache (`~/.cache/huggingface` or `$HF_HOME`) |
| Qwen3-VL-4B text encoder | Auto-downloaded from HuggingFace (`Qwen/Qwen3-VL-4B-Instruct`) on first run | HF cache |
| Chroma1-HD base weights (~8.9B, 643 tensors, native flow keys) | `hf download lodestones/Chroma1-HD` — or symlink a local copy | `chroma_checkpoint` in `chroma/configs/*.json` (default `checkpoints/chroma/Chroma1-HD.safetensors`) |
| Radiance x0 patch-16 base weights (~9.5B, 659 tensors, native flow keys) | Local file — symlink the trained base (see below) | `radiance_checkpoint` in `radiance/configs/*.json` (default `checkpoints/radiance/latest_x0.safetensors`) |
| T5-XXL text encoder + tokenizer (shared by chroma and radiance) | Auto-downloaded from HuggingFace (`lodestones/Chroma1-HD`, subfolders `text_encoder` / `tokenizer`) on first run | HF cache. Offline alternative: set `encoder_config` to `t5_xxl_local`, which reads `/mnt/datapool_u2/models/radiance/chroma/{text_encoder_2,tokenizer_2}` |
| Booru tag vocabulary (only for K2 tag-embedding runs) | Built locally, see below | `tag_embed.vocab_path` in `krea2/configs/*.json` (default `checkpoints/tag_vocab/tags_v1.parquet`) |

## Layout convention

```
checkpoints/
├── README.md          # this file (tracked)
├── krea2/
│   └── raw.safetensors            # K2 DiT base weights (bring your own)
├── chroma/
│   └── Chroma1-HD.safetensors     # Chroma1-HD base weights
└── radiance/
    └── latest_x0.safetensors      # symlink -> /mnt/datapool_u2/models/radiance/latest_x0.safetensors
```

### Radiance base weights

The patch-16 x0 base lives outside the repo and is symlinked in:

```bash
mkdir -p checkpoints/radiance
ln -sfn /mnt/datapool_u2/models/radiance/latest_x0.safetensors \
        checkpoints/radiance/latest_x0.safetensors
```

Pick the right file: `latest_x0.safetensors` is **patch 16** (659 tensors,
9.506B params). Its sibling `current_x0_x32.safetensors` is 14,155,832 bytes
larger — that delta is the bigger `img_in_patch` of a **patch-32** model, which
`RADIANCE_CONFIGS["radiance_x0_p16"]` will not load.

### Tag vocabulary (K2 tag embedding)

`krea2/train.py` with a `tag_embed` config block needs a tag list. Build it
from the LARGEST booru corpus available so the id space is complete and stable,
and point `--train-corpus` at what will actually be trained on so the builder
can report how much of the table will ever see a gradient:

```bash
uv run python utils/tag_vocab.py \
  --corpus       /mnt/datapool_u2/lodestone/caption_workspace/output/artist_samples \
  --train-corpus /mnt/datapool_u2/lodestone/caption_workspace/output/tag_samples_clean \
  --min-count 1 --out checkpoints/tag_vocab/tags_v1.parquet
```

The current `tags_v1.parquet` is 315,966 tags over 1.16M booru rows. **Never
overwrite one.** Tag ids are positions in a specific file, so regenerating over
a rebuilt corpus renumbers them and silently invalidates every trained table —
write `tags_v2.parquet` instead. The row count is stored in the checkpoint as
`tagembed.vocab_fingerprint` and asserted at load, so a mismatch raises rather
than trains against shifted meanings.

## Run outputs

Training runs write their own checkpoints to `runs/<run-name>/ckpts/`
(`lora_step_N.safetensors` or `full_step_N.safetensors`; TDM runs write
`tdm_student_step_N.safetensors` plus a resumable
`tdm_state_step_N.safetensors`). The matching `inference.py` consumes them
directly via `--lora-checkpoint`, or via `--mmdit-checkpoint` /
`--chroma-checkpoint` / `--radiance-checkpoint` for a merged full model. TDM
student checkpoints use the standard LoRA key convention, so
`--lora-checkpoint ... --steps 4 --guidance 0` runs them as-is.

Note: every `*_checkpoint` key in a config accepts any path — absolute paths to
checkpoints elsewhere on disk work fine.

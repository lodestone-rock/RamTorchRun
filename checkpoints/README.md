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
| MiniMax-H3 components (diffusers format, ~135 GB total) | `hf download MiniMaxAI/MiniMax-H3 --include "transformer/*" ...` per component, then symlink (see below) | `checkpoints/minimaxh3/{transformer,text_encoder,vae,audio_vae}` |

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

### ConvRot W8A8 (INT8) K2 checkpoints

`krea2/tools/quantize_convrot_w8a8.py` turns a K2 DiT checkpoint into stock
ComfyUI's native `int8_tensorwise` + ConvRot format (ComfyUI >= 0.27, no
custom node) — the same layout as Clybius's `kroma-*-W8A8.safetensors`:

```bash
uv run python krea2/tools/quantize_convrot_w8a8.py checkpoints/opd_base_candidate.safetensors
# -> checkpoints/opd_base_candidate_W8A8_convrot.safetensors (~13.5 GB, from 26.3 GB)
```

The 224 block linears become `weight` (int8, pre-rotated by a group-256
regular Hadamard) + `weight_scale` (f32 `[out, 1]`) + `comfy_quant` (JSON
marker); everything else keeps its source dtype. `krea2/inference.py` detects
the markers and runs it single-GPU (no `--offload`/`--pipeline`, no LoRA —
merge first, then quantize). `krea2/tools/compare_convrot_w8a8.py` measures it
against the BF16 source.

### MiniMax-H3 weights

The H3 port (`minimaxh3/`) consumes the diffusers-format repo
`MiniMaxAI/MiniMax-H3`, four components (~135 GB total):

```bash
for sub in transformer text_encoder vae audio_vae; do
  hf download MiniMaxAI/MiniMax-H3 --include "$sub/*" \
    --local-dir "checkpoints/minimaxh3-raw/$sub"
done
```

then either point `--ckpt-root` at the raw dir or symlink so the default
layout `checkpoints/minimaxh3/{transformer,text_encoder,vae,audio_vae}`
resolves. The transformer is ~33B params in mixed precision (bf16 blocks,
fp32 input/output projections + time embedder — `minimaxh3/model/dit.py`'s
`from_pretrained` preserves per-tensor dtypes, do NOT cast the whole thing).
The text encoder is Qwen3-VL-32B in bf16; the port truncates it to the first
50 layers at load (the checkpoint's own convention — H3 reads the layer-50
hidden state).

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

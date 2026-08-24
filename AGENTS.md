# Agent guide for RamTorchRun

This repo is a set of **standalone, runnable RamTorch examples** built around
large diffusion models (Krea-2 ~12B, Chroma1-HD 8.9B, Radiance x0 9.5B):
pipeline-parallel training, single-GPU offload training, and offloaded /
pipelined inference on a single node with multiple GPUs and **no NVLink**.

## Read this first

1. `memory/project_overview.md` — what this repo is for, current state, roadmap.
2. `memory/ramtorch_notes.md` — the RamTorch APIs used here and where the deep docs live.
3. `memory/worklog.md` — append-only log of what was changed and why.

## Conventions for agents

- **Update `memory/worklog.md`** at the end of any session that changes code:
  date, what changed, why, and any gotchas discovered. Keep entries short.
- If you change project direction, architecture, or scope, also update
  `memory/project_overview.md` so the next agent starts from the truth.
- This repo must stay **standalone**: never import from the symlinked
  `x0-pred/` or `RamTorch/` reference repos. They exist only for reading
  (original implementation and library source/docs). Both are gitignored.
- RamTorch is consumed as a **PyPI dependency** (`ramtorch>=1.6.4`), managed
  by uv (`uv add` / `uv sync`, run scripts with `uv run python ...`).
- Weights and run artifacts live in gitignored `checkpoints/` and `runs/`;
  see `checkpoints/README.md`.

## One folder per model, zero cross-model imports

The repo is organized **per model**, not per concern. Each model folder owns
its trainers, inference script, model code, and configs; only generic infra
(`dataloaders/`, `utils/`) is shared. A model folder must never import from
another model folder — copy and adapt instead. Adding e.g. Flux means creating
`flux/` with the same shape, not generalizing `krea2/`.

Every script carries a 2-line `sys.path` shim so both invocation styles work:
`uv run python krea2/train.py <cfg>` and `uv run python -m krea2.train <cfg>`.

## Dice once, choose the strategy with a flag

Each model is diced into a flat list of chunk modules (one per transformer
block) by its `model/chunks.py`. RamTorch executes that ONE list three ways,
selected by config (`parallelism`) or CLI flag rather than by separate code
paths: `offload` (1 GPU, weights streamed), `pipeline` (N GPUs, resident),
`pipeline-offload` (N GPUs, streamed). Adding an execution mode means adding a
flag, never a second trainer.

If you change the chunk dicing or the tuple relayed between chunks, run that
model's `tools/check_chunk_parity.py`. It compares forward output and every
gradient against the monolithic model across all execution modes, on CPU in
seconds, and catches the contract bugs (`out_no_grad`, grad-requiring float
leaves, tuple relay) that are expensive to find on GPU. Likewise run
`tools/check_tdm_roles.py` after touching `model/lora.py` or the role plumbing.

Note that `radiance/` relays a tuple that **changes shape** partway down the
chunk list, so it needs `set_resident_out_no_grad_per_stage` rather than
RamTorch's global `set_resident_out_no_grad`. If you add a model whose relay is
not uniform, reuse that helper.

## Repo map

Three model folders, all the same shape:

```
krea2/                    # Krea-2 ~12B MMDiT + Qwen-Image VAE + Qwen3-VL encoder
  model/                  #   MMDiT, VAE, text encoder, LoRA, sampling
  model/chunks.py         #   flat dicing: build_dit_chunks / build_encoder_chunks
  train.py                #   THE trainer: offload / pipeline / pipeline-offload, LoRA or full
  train_tdm.py            #   TDM few-step distillation (role-based LoRA)
  train_utils.py          #   K2 helpers (VAE encode/decode, timesteps, SDPA pinning)
  inference.py            #   single-GPU / --pipeline / --offload (combinable)
  tools/                  #   check_chunk_parity.py — chunked vs monolithic, CPU, seconds
  configs/                #   train_{offload,pipeline,pipeline_offload}_{lora,full}.json + train_smoke.json
chroma/                   # Chroma1-HD 8.9B flux-style DiT + T5-XXL + flux VAE
                          #   59 chunks; same file layout as krea2/
radiance/                 # Radiance x0 patch-16, 9.5B: Chroma in PIXEL space, NO VAE
  model/                  #   + NerfEmbedder / NerfGLUBlock / NerfFinalLayerConv, no autoencoder.py
  model/chunks.py         #   64 chunks; the relay CHANGES SHAPE at the NeRF head
  tools/                  #   + check_txt_pos_ids.py — scores a checkpoint's own
                          #     loss to settle the text-RoPE convention
dataloaders/              # SHARED: parquet image+caption dataset, aspect-ratio bucketing
utils/
  checkpoint.py           # SHARED: LoRA checkpoint load / merge helpers
  profiling.py            # SHARED: Perfetto trace capture over a sampling loop
  ramtorch_helpers.py     # SHARED: grad flush / accumulator plumbing for Pipeline
checkpoints/              # weights (gitignored, see its README)
memory/                   # agent context files (tracked)
runs/, profiles/          # training artifacts / profiler traces (gitignored)
RamTorch/, x0-pred/       # symlinked reference repos (read-only, gitignored)
```

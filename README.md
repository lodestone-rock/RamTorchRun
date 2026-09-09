# RamTorchRun

Standalone, runnable examples of [RamTorch](https://pypi.org/project/RamTorch/)
on a single node with multiple GPUs — **no NVLink required**.

Three demo models, one folder each, all the same shape:

| folder | model | notes |
|---|---|---|
| `krea2/` | Krea-2, ~12B MMDiT | Qwen-Image VAE + Qwen3-VL-4B encoder |
| `chroma/` | Chroma1-HD, 8.9B flux-style DiT | T5-XXL + flux VAE, distilled modulation |
| `radiance/` | Radiance x0 patch-16, 9.5B | **pixel space, no VAE** — NeRF decoder head, predicts x0 |

Everything starts from the same idea: the model is **diced once** into a flat
list of chunk modules (one per transformer block), and how those chunks are
executed is a flag. That gives three hardware strategies from one code path,
for both training (`<model>/train.py`) and inference (`<model>/inference.py`):

| strategy | GPUs | weights live in | use it when |
|---|---|---|---|
| `offload` | 1 | CPU RAM (or NVMe), streamed | one small / partially-free GPU |
| `pipeline` | N | resident on their stage's GPU | N GPUs, model fits when split |
| `pipeline-offload` | N | CPU RAM, streamed per stage | N GPUs, and it still doesn't fit |

What the examples demonstrate:

1. **Inference with offloading** (`krea2/inference.py --offload`) — the whole
   bf16 DiT lives in CPU pinned memory (or on NVMe) and streams through a
   small GPU window (`OffloadModel`), so one small/partially-free GPU runs the
   full model with minimal speed regression.
2. **Pipeline-parallel inference** (`krea2/inference.py --pipeline`) — chunks
   split into per-GPU stages (~9 GB/GPU on 4 GPUs instead of ~35 GB on one),
   optionally `--pipeline --offload` for both at once.
3. **Training with offloading** (`krea2/train.py`, `parallelism: "offload"`) —
   LoRA or full fine-tune of the 12B DiT on a **single** GPU: DiT and text
   encoder both stream from CPU pinned memory, the optimizer runs on the CPU
   masters.
4. **Pipeline-parallel training** (`parallelism: "pipeline"` /
   `"pipeline-offload"`) — LoRA or full fine-tune split across GPUs over plain
   PCIe (`Pipeline.step()`, staggered 1B1F schedule).

Each model folder also carries `train_tdm.py`, a TDM few-step distillation
trainer (teacher / fake-score / student share one frozen base and differ only by
LoRA role, so there are no extra weight copies).

The commands and measurements below use `krea2/`; substitute `chroma/` or
`radiance/` freely — the CLI, config keys and strategy flags are identical. The
performance numbers are K2's and do not transfer directly.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Then bring your own K2 DiT base weights — see
[checkpoints/README.md](checkpoints/README.md). The Qwen VAE and Qwen3-VL
encoder download automatically from HuggingFace on first run.

## Inference

```bash
# Baseline: everything on one large GPU
uv run python krea2/inference.py --config krea2/configs/train_pipeline_lora.json --no-lora \
    --prompt "a cat astronaut floating in space, cinematic lighting" --seed 0

# Offloaded: same model on ONE small GPU (weights stream from CPU RAM)
uv run python krea2/inference.py --config krea2/configs/train_pipeline_lora.json --no-lora \
    --offload --offload-window 2 \
    --prompt "a cat astronaut floating in space, cinematic lighting" --seed 0

# Pipeline-parallel across 4 GPUs
uv run python krea2/inference.py --config krea2/configs/train_pipeline_lora.json --no-lora \
    --pipeline --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    --prompt "a cat astronaut floating in space, cinematic lighting" --seed 0

# Both: 4 GPUs of compute, each streaming its own slice (lowest VRAM per GPU)
uv run python krea2/inference.py --config krea2/configs/train_pipeline_lora.json --no-lora \
    --pipeline --offload --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    --prompt "a cat astronaut floating in space, cinematic lighting" --seed 0
```

Same seed produces the same noise in every mode, so outputs are directly
comparable — use identical prompt+seed with and without `--offload` to verify
the (minimal) speed difference and identical results.

Measured on 4x RTX PRO 6000 (PCIe, no NVLink), 1024px, 28 steps + CFG:
offload costs ~3% vs the fully-resident baseline at `--batch-size 4`
(~31.3 vs ~30.3 s/image) — the streaming fully overlaps with compute.

Whether it overlaps is a simple race between two per-step quantities: the
weight stream is a **constant** ~2.0 s (51 GB for the cond+uncond passes at
~26 GB/s), while compute scales with the batch. Profiling at window 2 shows
the crossover sitting at batch 2:

| `--batch-size` | GPU busy | GPU idle | verdict |
|---|---|---|---|
| 1 | 51% | 0.87 s/step | PCIe-bound: compute (0.93 s) can't cover the 1.8 s stream |
| 2 | 98% | 0.04 s/step | break-even: compute ~1.94 s vs stream ~1.93 s |
| 4 | 99% | 0.05 s/step | compute-bound, streaming free |

So batch 2 is already enough to hide the transfers at 1024px; batch 1 is the
worst case.

If you must run batch 1, `--offload-pin` keeps N chunks permanently resident
and removes them from the per-step stream, trading VRAM for PCIe traffic.
Pinning half the model fully recovers batch-1 throughput:

| batch 1, model in 16 chunks | GB/step | GPU busy | wall/step | speedup |
|---|---|---|---|---|
| `--offload-pin 0` (0 of the model) | 48.5 | 51% | 1.80 s | 1.00x |
| `--offload-pin 4` (1/4) | 34.3 | 71% | 1.28 s | 1.41x |
| `--offload-pin 8` (1/2) | 24.3 | **99%** | 0.96 s | **1.88x** |

At half the model pinned the stream costs 0.92 s/step against 0.94 s of
compute — just covered. Peak DiT weight VRAM is `(window + pin)/chunks` of
the model, so pinning 8 of 16 chunks with window 2 holds
10/16 x 25.6 GB ~ 16 GB resident.

These numbers were measured with the DiT diced into 16 chunks. The default is
now **one chunk per block** (30 chunks), so pin counts scale accordingly —
`--offload-pin 15` pins the same half of the model. Use `--blocks-per-chunk`
to go coarser (fewer, larger chunks: less per-chunk overhead, coarser window
granularity).

A third tier, `--offload-nvme N --offload-nvme-path FILE`, keeps N chunks'
masters on disk instead of in CPU RAM (they stream disk -> pinned staging ->
GPU). With `--offload-pin 8 --offload-nvme 8` the two sets are exactly
complementary — even chunks on the GPU, odd chunks on disk — so **no DiT
master weights sit in host RAM at all**. That costs speed:

| batch 1, pin 8 | source | GPU busy | wall/step |
|---|---|---|---|
| `--offload-nvme 0` | CPU pinned RAM | 99% | 0.96 s |
| `--offload-nvme 8` | NVMe (mmap) | 51% | 1.91 s |

The GPU-side copy is unchanged (~24 GB/step either way); the disk read adds
~1.6 s/step of stall, roughly halving throughput. Use the NVMe tier when host
RAM is the binding constraint, not for speed. Note `--offload-nvme-path` must
be on a real drive — `/tmp` is often tmpfs (RAM), which silently defeats the
point. The scratch file is deleted when the model closes.

For prompt lists, use
`--prompts-file` and optionally fan out over GPUs with `--num-shards/--shard`
(one process per GPU, global seeds preserved).

### Profiling a run (Perfetto)

Both RamTorch inference modes can dump a Chrome/Perfetto trace of a few
diffusion steps:

```bash
# Offload: weight streaming next to compute on one GPU
uv run python krea2/inference.py --config krea2/configs/train_pipeline_lora.json --no-lora \
    --offload --batch-size 4 --steps 8 \
    --profile profiles/offload.json --profile-warmup 1 --profile-steps 3 \
    --prompt "..." --prompt "..." --prompt "..." --prompt "..."

# Pipeline: per-GPU stage overlap across 4 GPUs
uv run python krea2/inference.py --config krea2/configs/train_pipeline_lora.json --no-lora \
    --pipeline --devices cuda:0 cuda:1 cuda:2 cuda:3 --batch-size 4 --steps 8 \
    --profile profiles/pipeline.json --profile-warmup 1 --profile-steps 3 \
    --prompt "..." --prompt "..." --prompt "..." --prompt "..."
```

`--profile-warmup` steps run before capture starts (keeping allocator growth,
cuDNN autotuning and the first cold weight stream out of the trace), then
`--profile-steps` steps are recorded and the run stops — it writes a trace,
not images. Drop the file into [ui.perfetto.dev](https://ui.perfetto.dev);
gzip it first (`gzip trace.json`, ~13x smaller) since Perfetto opens `.gz`
directly.

Whenever weights stream, the trace also carries an **`offload h2d loader`**
track spliced in from RamTorch's worker thread, so you can see each chunk's
H2D copy running ahead of the `F{k} infer` compute span that consumes it, with
`wait L{k}` marking any stall. Under `--pipeline --offload` there is one such
track per stage, prefixed `s0 `, `s1 `, ...

Note when reading these: the CPU-side `wait L{k}` spans stay near zero even
when the GPU is starving, because a chunk counts as resident once its copy is
*enqueued* — the compute stream then waits on a CUDA event, which is a
device-side stall invisible to CPU annotations. To judge overlap, measure GPU
busy time (sum of `kernel` event durations) against the `diffusion_step_{i}`
wall span, or just look for gaps in the GPU track.

## Training

One trainer, `krea2/train.py`, covers every combination of fine-tuning target
(`mode`: `lora` / `full`) and hardware strategy (`parallelism`: `offload` /
`pipeline` / `pipeline-offload`). Edit the config first — dataset and
checkpoint paths — then:

```bash
# LoRA on ONE GPU, weights streaming from CPU RAM
uv run python krea2/train.py krea2/configs/train_offload_lora.json

# Full fine-tune on ONE GPU (fp32 CPU masters + bf16 compute, ~255 GB host RAM)
uv run python krea2/train.py krea2/configs/train_offload_full.json

# LoRA over 4 GPUs, weights resident on their stage
uv run python krea2/train.py krea2/configs/train_pipeline_lora.json

# Full fine-tune over 4 GPUs, resident
uv run python krea2/train.py krea2/configs/train_pipeline_full.json

# 4 GPUs AND streaming — for when the resident pipeline OOMs
uv run python krea2/train.py krea2/configs/train_pipeline_offload_lora.json
uv run python krea2/train.py krea2/configs/train_pipeline_offload_full.json

# Quick smoke run (256px, 6 steps) — any strategy, chosen on the command line
uv run python krea2/train.py krea2/configs/train_smoke.json \
    --parallelism pipeline-offload --devices cuda:1 cuda:3
```

Plain `python` — a single process drives all pipeline stages (no torchrun).
Effective batch size = `batch_size` (per microbatch) × `n_microbatches`;
microbatches serve as gradient accumulation in every strategy. Checkpoints,
previews, and a CSV loss log land in `runs/<name>/`. Training data comes from
a parquet file listing image paths, captions, and dimensions (see the
`parquet_dataloader` block in the configs).

Wherever weights stream, the 12.8B DiT never fully resides on the GPU: chunks
move from CPU pinned memory through an `offload_window` of GPU slots in both
the forward and backward direction. The frozen Qwen3-VL encoder streams the
same way; only the small VAE stays resident. In full fine-tuning gradients
accumulate on the GPU (`offload_grad_accum: "stream"`) and cross PCIe once per
step at flush, after which AdamW updates the CPU masters with the fused CPU
kernel; LoRA configs use `"cpu"` instead (see the knobs below).

Measured on one RTX PRO 6000 (single-GPU offload, 256px, effective batch 4,
16 chunks, `offload_window: 4`):

| mode | step time | host RAM | bottleneck |
|---|---|---|---|
| `lora` (bf16 masters) | ~4.4 s | ~30 GB | compute; streaming fully hidden |
| `full` (fp32 masters) | ~35 s | ~256 GB | CPU AdamW + grad flush over 12.8B params |

In both cases `acquire_wait_s` stays near zero, i.e. the weight streaming is
completely overlapped — full FT is limited by CPU-side work, not by PCIe.

**For full FT, stream weights only if they do not fit.** On 4 GPUs at 256px
(batch 2 x 2 microbatches), the same step costs 6.4 s with resident stages and
30.0 s with streamed ones. The trainer's `Time split:` line shows why: with
streamed weights, 61% of the step is the host touching 51 GB of fp32 masters
and gradients (grad flush 34%, `clip_grad_norm_` 13%, CPU AdamW 14%), against
~0% resident, where the fused CUDA AdamW and a GPU-side norm are effectively
free. It is not a smooth dial either — the moment a chunk's master sits on the
CPU it pays that host cost, so streaming just 1-2 chunks per GPU already costs
60% to save ~5 GB. Streaming buys headroom (18.6 GB peak vs 68.7 GB), which is
what makes 12.8B full FT possible on small cards at all; it does not buy speed.

Knobs worth reaching for, in order:

- `offload_window` — deeper prefetch; `offload_pin` — chunks kept resident,
  trading VRAM for PCIe traffic.
- `offload_backward` — `"checkpoint"` (per-chunk recompute, cheapest VRAM,
  dropout-safe) or `"keep"` (fastest, holds every chunk's graph).
- `offload_grad_accum` — follow `mode`, not the parallelism. Measured on K2 at
  256px: `"stream"` is 26% faster than `"cpu"` for full FT (30.0 vs 40.7 s/step)
  and 22% *slower* for LoRA (7.5 vs 5.8), where the trainable slice is small
  enough that host-side adds are cheap while the GPU accumulator slots thrash.
- `offload_activations` — stream saved activations to pinned CPU too. Only
  worth it with `"keep"`; under `"checkpoint"` the engine already caches just
  the chunk boundaries. On the LoRA smoke it took `"keep"` from 26.3 GB to
  10.9 GB peak for +12% step time. It does NOT help full FT: there the memory
  is weights + grads + Adam state (51 GB/GPU of a 68 GB peak), not activations,
  and offloading them under `"checkpoint"` even adds ~4 GB of staging.
- `optimizer` — `"adamw"` (fused; CPU kernel for streamed masters, CUDA kernel
  for resident ones) or `"offload-adamw"`, RamTorch's state-in-host-RAM AdamW
  sharded one per stage. The latter measured *worse* on K2 full FT (41.3 vs
  30.5 s/step): one pass over host DDR beats streaming 28 B/param over PCIe,
  even with four links in parallel. It is here for models whose optimizer state
  does not fit in host RAM at fp32.
  Krea-2's ordinary `train.py` additionally supports `"muon"` for resident
  `parallelism: "pipeline"`, `mode: "full"` training. It uses native
  `torch.optim.Muon` for hidden linear weights and AdamW for the remaining
  trainable parameters; tag embeddings retain their separate RowAdamW.
  LoRA, streamed modes, TDM and mass-LoRA are not supported by this option.
  Both optimizer families follow the existing warmup. This is standard Muon,
  not chunked Muon (CMuon). Krea ordinary `train.py` also supports `"lion"`
  for both full FT and LoRA, resident or streamed. Lion updates the normal
  trainable list without matrix-specific routing; tags keep RowAdamW.
- `grad_ckpt` — per-block `torch.utils.checkpoint`, **resident stages only**.
  Under streaming it would recompute against the CPU masters; use
  `offload_backward: "checkpoint"` instead.

### Krea-2 Muon learning rates

Set `optimizer: "muon"` to opt in; existing AdamW configurations are unchanged.
The default `muon.adjust_lr_fn: "match_rms_adamw"` uses the Moonlight scaling
`0.2 * sqrt(max(rows, columns))`, so `lr` is on an AdamW-like scale rather than
original Muon's scale. The fallback AdamW learning rate defaults to `lr`.
Optional `muon` keys are `momentum` (0.95), `nesterov` (true), `ns_steps` (5),
`adjust_lr_fn` (`"match_rms_adamw"`), `adamw_lr` (defaults to `lr`) and
`exclude_prefixes` (dotted module/parameter names routed to AdamW instead).
For example, add this fragment to a resident full-FT config:

```json
{
  "optimizer": "muon",
  "lr": 5e-6,
  "muon": {
    "adjust_lr_fn": "match_rms_adamw",
    "momentum": 0.95,
    "ns_steps": 5
  }
}
```

This is update-scale matching, not a guarantee that either optimizer has the
same optimal LR. See the [PyTorch Muon documentation](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Muon.html).

Diffusion evidence: [CMuon (2026), sections 4 and 5.4](https://arxiv.org/html/2608.02502)
uses global batch 1024 and sweeps 1e-4, 2e-4 and 3e-4; 2e-4 is near-optimal
for its ImageNet DiT-XL training experiments. Those results use **chunked**
Muon and training from scratch, not vanilla Muon fine-tuning of Krea-2.
They do not establish a good LR for a 12.8B checkpoint at effective batch 16.
For the existing low-LR annealing recipe, a conservative first experiment is
RMS-matched Muon at the existing 5e-6, then a controlled comparison around it
(e.g. 2.5e-6, 5e-6, 1e-5). This is an unvalidated recommendation, not a measured
optimum. Keep effective batch fixed across resolutions; no automatic LR change
or batch-scaling rule is applied when the microbatch/accumulation changes.
Muon replaces two Adam moments with one momentum buffer for its matrix subset,
but Newton–Schulz adds matrix multiplications and temporary memory; lower state
memory does not guarantee faster steps or lower peak VRAM. Existing checkpoint
files still save weights only, not Muon/AdamW moments or scheduler state.

### Krea-2 preview microbatching

Ordinary `krea2/train.py` accepts `preview_samples` as a positive integer or
`"microbatches"` (the current step's accumulation count), capped to the available
training batch. Preview text encoding and each DiT CFG pass use microbatches
bounded by the current training microbatch size; VAE decoding uses the same
bound and moves each decoded chunk to CPU. For E32/b1/M32,
`"preview_samples": 32` schedules 32 size-one microbatches per CFG pass, not one
32-image batch or 32 independent sampling loops. `eval_interval` is unchanged.
Optional `preview_n_microbatches` independently caps the number of microbatches
per preview encoder/DiT inference call. Set it to `16` with 32 samples/b1 to run
two groups of 16, without changing training accumulation or increasing the
per-microbatch image count. Omit it (or use null) for the previous uncapped behavior.

### Krea-2 transformer-block compilation

Ordinary `krea2/train.py` supports opt-in `"compile": true` for resident
`parallelism: "pipeline"`. It lazily compiles individual DiT transformer blocks
(including text-fusion blocks) and the executed Qwen text decoder layers.
The VAE, embeddings, projections, output heads, and pipeline orchestration
remain eager. Parameter identities and checkpoint keys are preserved.

The equivalent explicit configuration is:

```json
{
  "compile": {
    "dit": true,
    "encoder": true,
    "backend": "inductor",
    "mode": "default",
    "dynamic": true,
    "fullgraph": false
  }
}
```

CUDA graphs are disabled for the threaded pipeline. Block forwards share
Dynamo's compilation lock and force backward lowering there to avoid PyTorch
2.10's process-global FX tracing race. This serializes Python forward dispatch,
not asynchronous GPU kernels; throughput must be measured. Persistent and
mixed-order reductions are disabled: Torch 2.10 mixed-order fusion otherwise
forces a persistent RMSNorm backward kernel over the GPU shared-memory limit.
Dynamic shapes accommodate
varying caption lengths/aspect buckets but do not guarantee zero recompiles.
Compilation is lazy: first-use training/backward and preview paths can be slow.
Backend errors are not silently swallowed; `fullgraph: false` can still allow
normal Dynamo graph breaks. `aot_eager`/`eager` backends are diagnostic options,
not production speedups. Weight streaming is unsupported with compilation.
See [PyTorch regional compilation guidance](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html).

### Krea-2 Lion learning rates

Lion is a shape-independent replacement for the main optimizer in ordinary
`krea2/train.py` (`mode: "full"` or `"lora"`; all three parallelisms).
It keeps one momentum buffer per updated parameter and uses sign updates;
no Muon/AdamW matrix split is needed. Tag embeddings still use RowAdamW to
avoid updating untouched rows. TDM and mass-LoRA trainers are not extended.
This is a straightforward tensor implementation, not a fused/Triton kernel;
streamed masters and their Lion state stay on CPU. No GPU performance claim.

The [Lion authors, section 5](https://arxiv.org/abs/2302.06675) recommend
**3–10x smaller LR and 3–10x larger weight decay** than AdamW, keeping the
product `lr * weight_decay` similar. Their diffusion example uses AdamW
`lr=3e-4, weight_decay=0.01` versus Lion `lr=3e-5, weight_decay=0.1`.
They scale the whole LR schedule by the same ratio, retaining the schedule
shape and clipping. This is a tuning heuristic, not a universal optimum;
the paper reports smaller advantages at batch sizes below 64.

For this repository's anneal baseline (`lr=5e-6, weight_decay=1e-4`), the
10x conversion is the following config fragment, **not an automatically
applied change**:

```json
{
  "optimizer": "lion",
  "lr": 5e-7,
  "weight_decay": 1e-3,
  "lion": {"betas": [0.9, 0.99]}
}
```

The `lion` block is optional; its only option is `betas` (default `[0.9, 0.99]`).
If omitted, Lion's top-level LR defaults to `1e-5`, and weight decay to `1e-4`;
set both explicitly when converting a known AdamW recipe.
Explicit LR/weight decay are used literally, with no automatic /10 or x10
conversion. Existing warmup is retained; effective batch remains constant
across resolutions, so LR is not changed when accumulation changes. Explicit
`tag_embed.lr` and tag weight decay are separate settings and are not converted;
if tag LR is omitted, it inherits the resolved top-level LR (including Lion's
`1e-5` default). Set it explicitly to preserve the previous RowAdamW learning
rate when switching to Lion. Checkpoints
remain weights-only, with fresh momentum/scheduler on resumed training.

Before spending GPU hours on a change to the chunk dicing, run the parity
harness — it checks the chunked forward AND every gradient against the
monolithic model, across all execution modes, on CPU in seconds:

```bash
uv run python krea2/tools/check_chunk_parity.py     # or chroma/, radiance/
uv run python krea2/tools/check_tdm_roles.py        # after touching LoRA roles
```

`radiance/` has one extra tool. Its text RoPE convention (`txt_pos_ids`) differs
between the two upstream trainers, and both settings render plausible images, so
it is settled by scoring a checkpoint against its own training objective rather
than by eye:

```bash
uv run python radiance/tools/check_txt_pos_ids.py --config radiance/configs/train_smoke.json
```

## Repo layout

One folder per model; only generic infrastructure is shared, and model folders
never import from each other.

```
krea2/                 # Krea-2: owns its trainer, inference, model code, configs
  model/               #   MMDiT, VAE, text encoder, LoRA, sampling
    chunks.py          #   flat chunk dicing: build_dit_chunks / build_encoder_chunks
  train.py             #   trainer: offload / pipeline / pipeline-offload, LoRA or full
  train_tdm.py         #   TDM few-step distillation (role-based LoRA)
  train_utils.py       #   K2 helpers (VAE encode/decode, timesteps, SDPA pinning)
  inference.py         #   single-GPU / --pipeline / --offload (combinable)
  tools/               #   check_chunk_parity.py: chunked vs monolithic, CPU, seconds
  configs/             #   train_{offload,pipeline,pipeline_offload}_{lora,full}.json
chroma/                # Chroma1-HD, same layout (59 chunks)
radiance/              # Radiance x0 patch-16, same layout (64 chunks), NO autoencoder
dataloaders/           # shared: parquet dataset with aspect-ratio bucketing
utils/
  checkpoint.py        # shared: LoRA checkpoint load / merge helpers
  profiling.py         # shared: Perfetto trace capture over a sampling loop
  ramtorch_helpers.py  # shared: grad flush / accumulator plumbing for Pipeline
checkpoints/           # weights (gitignored; see its README)
memory/                # agent context: overview, RamTorch notes, worklog
runs/                  # training output: checkpoints, previews, loss CSV (gitignored)
profiles/              # profiler traces (gitignored)
```

Every script runs either way — `uv run python krea2/train.py <cfg>` or
`uv run python -m krea2.train <cfg>`.

This is an agent-friendly workspace: see [AGENTS.md](AGENTS.md) and the
[memory/](memory/) folder for project context and conventions.

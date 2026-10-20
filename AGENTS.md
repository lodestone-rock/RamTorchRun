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

`krea2/` can also inject **tag embeddings**: a learnable per-tag table
(`model/tag_embed.py`) whose matched tags become extra tokens on the DiT
sequence between the text prefix and the image tokens, so a prompt can mix
booru bag-of-words tags with natural language. It is off unless a config
carries a `tag_embed` block, and an untagged row is a bit-exact no-op. The
policy lives in one place — `train_utils.TagTrainer`, shared by both trainers.
Run `krea2/tools/check_tag_embed.py` after touching it. Two rules that are easy
to break: the table is **attached after** the base checkpoint loads (the base
has no tag keys, so it cannot go in the config), and `tag_ids`/`tag_mask` sit
**immediately after `mask`** in `SingleStreamDiT.forward` so the monolithic
model and `DiTEmbedChunk` take the same positional tuple.

`krea2/` has a fourth way of using that same chunk list: **mass LoRA**
(`train_mass_lora.py`), which trains L per-concept adapters at once by stacking
each `nn.Linear`'s adapter into a bank and routing a slot-packed batch through
a grouped `bmm`. It needs no change to the dicing or the relay — only per-step
module state. Run `krea2/tools/check_lora_bank.py` after touching
`model/lora_bank.py`, `utils/bank_optimizer.py`, or the slot packing; it proves
on CPU that the bank equals L separate LoRA models and that inactive slots move
by exactly zero.

Mass LoRA also inverts one of the shared dataloader's assumptions. The parent
picks ONE caption column per row at load time, which is a fair sample of the mix
across a million rows; a mass-LoRA slot holds ~5 images and revisits each ~250
times, so that would pin every image to one band for a whole run.
`mass_lora_dataloader.py` therefore keeps every band and draws one per VISIT —
run `dataloaders/check_mass_lora_captions.py` after touching captions there. It
also means the plan is only reproducible while RNG *consumption* in the row loop
is unchanged: adding or removing a per-row `rng` call re-rolls the plan, which
invalidates the `parquet_dataloader.offset` trick for resuming mid-plan.

`moss/train_sft.py` adds a batched forward to `model/transformer.py` (right-padded
text, bool key mask, per-sample RoPE; `seq_pad` pads the whole sequence to a
multiple so torch.compile sees few static shapes). Run
`moss/tools/check_batched_forward.py` after touching the forward, and
`moss/tools/check_sft_recipe.py` after touching `utils/muon.py`, the optimizer
partition, `sample_t` or the EMA. Under compile, keep
`torch._inductor.config.triton.mix_order_reduction = False` (torch 2.10's fused
RMSNorm backward overruns shared memory at ~4.6k+ tokens per micro-batch).

On this box (4x 96 GB PCIe, two sockets, no NVLink) run it with `parallelism:
"zero"` (`utils/zero_sharding.py`, ramtorch `MultiGPUWrapper`'s ZeRO-1 ported to
torchrun): each parameter is owned whole by one rank, which alone holds its
Muon/AdamW state and EMA shadow; grads are reduced to owners once per update,
owners step and broadcast. Run `moss/tools/check_zero_sync.py` after touching
it. Two traps: ZeRO checkpoints must use `flatten_optimizer_state_dict=True`
(`ZERO_STATE_OPTIONS`) or the list-indexed `param_groups` keys collide across
ranks; and NCCL must get `NCCL_P2P_LEVEL=SYS` (train_sft sets it) — its default
stages cross-socket traffic through host memory at ~3.4 GB/s instead of ~18
(`moss/tools/probe_comms.py`).

`moss/train_concept_lora.py` is the batch-1 artist LoRA trainer (one GPU, 1024 px,
driven by `moss/tools/launch_artist_loras.py`). Its speed comes from switches
measured by `moss/tools/bench_concept_lora.py`: the frozen text projection is
cached (`forward(..., projected=True)`), blocks are compiled, q/k/v run as one
GEMM, and the frozen base matmuls run in FP8 (`model/fast_lora.py`, rowwise
`torch._scaled_mm` as `torch.library` custom ops — an `autograd.Function` there
trips dynamo when two projections share an input). Result: 0.75 -> 0.39 s/step,
51 -> 34.5 GiB. Run `moss/tools/check_concept_lora.py` (add `--gpu` for FP8) after
touching the trainer, `fast_lora.py` or the projected-text path.

`krea2/train_opd.py` distills every trained artist LoRA (`runs/k2-artist-loras`)
into ONE student LoRA, DiffusionOPD/DanceOPD style, on torchrun + the same
`ZeroSync` (student params only). `model/teacher_bank.py`'s `DistillLinear`
holds the frozen base, the student LoRA and an S-slot teacher bank in buffers;
one DiT serves as base / student / teacher (grouped bmm, row group i = slot i)
via `set_distill_mode`, or `teacher.base: "separate"` loads a second DiT.
`dataloaders/lora_teacher_loader.py` scans the teachers, partitions them over
ranks, and feeds pinned LoRAs from an LRU; the bank is refilled each micro-step
on a side stream. `policy.off_policy_updates` of dataset x_t first, then a
mix with student rollouts (`query`: dense | subset | single_low_noise). Run
`krea2/tools/check_teacher_bank.py` after touching the bank, the slot loader or
the teacher loader. Student rank must be uniform (`merge_lora_into_base_sd`).
Max rank on 4x96 GB (full grad ckpt, 4 rows/micro, `tools/sweep_opd_rank.sh`):
shared ~1280 @512 / ~1152 @1024; separate ~736 @512 / ~576 @1024.

## Repo map

Three model folders, all the same shape:

```
krea2/                    # Krea-2 ~12B MMDiT + Qwen-Image VAE + Qwen3-VL encoder
  model/                  #   MMDiT, VAE, text encoder, LoRA, sampling
  model/chunks.py         #   flat dicing: build_dit_chunks / build_encoder_chunks
  model/tag_embed.py      #   TagEmbedder: per-tag table -> extra DiT tokens
  model/convrot.py        #   ConvRot W8A8: regular-Hadamard INT8 linear, ComfyUI
                          #   int8_tensorwise format (tools/quantize_convrot_w8a8.py
                          #   writes it, tools/check_convrot.py guards it,
                          #   tools/compare_convrot_w8a8.py measures it)
  train.py                #   THE trainer: offload / pipeline / pipeline-offload, LoRA or full
  train_tdm.py            #   TDM few-step distillation (role-based LoRA)
  train_mass_lora.py      #   L per-concept LoRAs at once (stacked banks + grouped bmm)
  train_e2e_qwen.py       #   E2E "pseudo-VAE", DDP: LoRA on DiT + Qwen3-VL (LM+vision),
                          #   alternating caption (image->scene_graph JSON) /
                          #   diffusion (DiT on Qwen(JSON), grad into Qwen) steps;
                          #   model/qwen_e2e.py + tools/check_e2e_qwen.py
  train_concept_lora.py   #   ONE concept: cached dataset, concept token, Lion
                          #   + warmup/constant/COOLDOWN. launcher + CPU check
                          #   in tools/launch_concept_lora.py, model/marker parser
  train_opd.py            #   all artist LoRAs -> ONE student LoRA (OPD), torchrun +
                          #   ZeroSync; model/teacher_bank.py (DistillLinear: base /
                          #   student / S-slot teacher bank), off -> on-policy flag,
                          #   teacher shared | separate base; configs/train_opd_*.json,
                          #   tools/check_teacher_bank.py (CPU), tools/sweep_opd_rank.sh
  train_utils.py          #   K2 helpers (VAE encode/decode, timesteps, SDPA pinning)
                          #   + TagTrainer: the whole tag-embedding policy
  inference.py            #   single-GPU / --pipeline / --offload (combinable), --tags
  tools/                  #   check_chunk_parity.py — chunked vs monolithic, CPU, seconds
                          #   + check_tag_embed.py — parity, no-op, permutation, matcher
  configs/                #   train_{offload,pipeline,pipeline_offload}_{lora,full}.json + train_smoke.json
                          #   + train_pipeline_lora_tags.json, train_tag_smoke.json
chroma/                   # Chroma1-HD 8.9B flux-style DiT + T5-XXL + flux VAE
                          #   59 chunks; same file layout as krea2/
radiance/                 # Radiance x0 patch-16, 9.5B: Chroma in PIXEL space, NO VAE
  model/                  #   + NerfEmbedder / NerfGLUBlock / NerfFinalLayerConv, no autoencoder.py
  model/chunks.py         #   64 chunks; the relay CHANGES SHAPE at the NeRF head
  tools/                  #   + check_txt_pos_ids.py — scores a checkpoint's own
                          #     loss to settle the text-RoPE convention
moss/                     # MOSS-Image-SFT 5B (Qwen3.5-9B encoder, FLUX.2 VAE), torchrun DDP:
                          #   train_dataspace_dmd.py (data-space DMD, role LoRA, t=1 folded
                          #   into the time pathway), train_dmad.py; tools/check_dataspace.py
                          #   + check_dmad.py (CPU); weights checkpoints/moss-image-sft
  train_sft.py            #   full SFT, flow matching; parallelism fsdp | ddp | zero; flags: batched
                          #     forward (padded text + key mask, seq_pad -> static shapes),
                          #     compile (static, blocks only), sdpa cudnn, grad_checkpoint N,
                          #     optimizer adamw | muon (report's hybrid Muon-AdamW, utils/muon.py),
                          #     timestep kind moss_final, EMA (export = EMA); ParquetTextImageDataset
                          #     (+ min_pixels, rustfs file_path); milestones export a release-layout
                          #     bf16 transformer/. configs/sft_kromoss_v1*.json: continue
                          #     lodestones/Kromoss (checkpoints/kromoss) under zero on 4 GPUs
  train_concept_lora.py   #   ONE artist LoRA on Kromoss, 1 GPU, batch 1 @1024 (krea2's port):
                          #     cached latents + text, projected text, compile + fuse_qkv + FP8
                          #     base (model/fast_lora.py); kohya save. configs/concept_lora_artist.json,
                          #     tools/launch_artist_loras.py (reuses k2 artist lists),
                          #     tools/check_concept_lora.py (CPU; --gpu adds FP8),
                          #     tools/bench_concept_lora.py (speed ladder), preview_concept_lora.py
  tools/                  #   + check_batched_forward.py / check_sft_recipe.py (CPU, SFT trainer)
                          #   + check_zero_sync.py (CPU gloo: zero == one process, DCP resume,
                          #     reshard to a new world) / probe_comms.py (NCCL timings, torchrun)
                          #   + probe_sft_throughput.py (1 GPU: batch x ckpt x compile x sdpa)
                          #   + plot_sft_loss.py, checkpoint_watchdog.py (prunes old trainer state)
                          #   + export_comfy.py: release transformer/ -> ComfyUI single file
                          #     (fused proj.qkv; == Comfy-Org's file on the SFT release)
                          #   + eval_prompts.py: --transformer, per-NFE --cfg, --shard/--shards
dataloaders/              # SHARED: parquet image+caption dataset, aspect-ratio bucketing
                          #   + caption_stream.py / caption_image_stream.py (moss streams;
                          #     configs expect a trainer_samples_v3 symlink in the repo root)
                          #   + tag_column: tag ids DECOUPLED from the caption
                          #   + lora_teacher_loader.py: TeacherPool / ArtistStepSampler /
                          #     pinned LoRACache / StepQueue for train_opd
                          #   + mass_lora_dataloader.py: slot-packed step plans,
                          #     caption band drawn PER VISIT, not per row
                          #   + check_mass_lora_captions.py: guards that draw
                          #   + probe_mass_lora_plan.py: scores a step plan with
                          #     no model/GPU/decode; --fast streams 20M rows in ~1min
utils/
  checkpoint.py           # SHARED: LoRA checkpoint load / merge helpers
  muon.py                 # SHARED: Muon (Nesterov, quintic Newton-Schulz, RMS-matched lr, fused-matrix splits,
                          #   optional bf16 momentum)
  zero_sharding.py        # SHARED: ZeroSync — ZeRO-1 over torchrun ranks (whole-tensor owners,
                          #   reduce-to-owner, global | local clip, broadcast, EMA swap)
  profiling.py            # SHARED: Perfetto trace capture over a sampling loop
  ramtorch_helpers.py     # SHARED: grad flush / accumulator plumbing for Pipeline
  bank_optimizer.py       # SHARED: BankAdamW — AdamW over a LoRA bank's ACTIVE slots only
  row_optimizer.py        # SHARED: RowAdamW — AdamW over an embedding's TOUCHED rows only
  tag_vocab.py            # SHARED: tag normalizer, versioned vocab builder, TagMatcher
experiments/              # standalone analysis probes (not per-model): merge math
  merge_core.py           #   extracted from x0-pred/merge_lora_bank.py (port, no import)
  merge_lora_bank.py      #   the bank-merge CLI, thin over merge_core
  merge_scaling.py        #   n-sweep: why normmatch survives 500+ pooled LoRAs
  synthetic_coherence.py  #   controlled validation of the coherence theory
checkpoints/              # weights (gitignored, see its README)
memory/                   # agent context files (tracked)
runs/, profiles/          # training artifacts / profiler traces (gitignored)
RamTorch/, x0-pred/       # symlinked reference repos (read-only, gitignored)
```

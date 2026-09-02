# Project overview

## Purpose

`RamTorchRun` demonstrates [RamTorch](https://pypi.org/project/RamTorch/) with
**real, usable examples** (not snippets) on a single node with multiple GPUs,
deliberately avoiding NVLink-dependent techniques. The first demo model is
Krea-2 (K2), a ~12B-parameter MMDiT text-to-image diffusion model with a
Qwen-Image VAE and a Qwen3-VL-4B text encoder. The second is Chroma
(`chroma/`, lodestones/Chroma1-HD, 8.9B flux-style DiT with T5-XXL). The third
is Radiance (`radiance/`, 9.5B x0-prediction sibling of Chroma that reads and
writes **pixels** through a NeRF decoder head — no VAE at all).

Three target use cases, all implemented:

1. **Inference with offloading** — run a big, compute-bound model on a small
   GPU by streaming bf16 weights from CPU pinned memory (`OffloadModel`),
   with minimal speed regression vs. fully-resident execution.
2. **Training with offloading** — LoRA or full fine-tune on a single GPU with
   the DiT and text encoder both streamed from CPU RAM.
3. **Multi-GPU pipeline parallelism** — train and infer across GPUs over PCIe
   (no NVLink) using RamTorch's single-process `Pipeline`, optionally with
   per-stage weight streaming on top.

## Structure: one folder per model

The repo is organized per model, not per concern. `krea2/`, `chroma/` and
`radiance/` each own their trainers, inference script, model code, and configs.
Only generic infra is shared:
`dataloaders/` (parquet image+caption dataset with aspect-ratio bucketing;
the scipy-based OT variant was intentionally not ported),
`utils/checkpoint.py` (LoRA load / merge helpers), `utils/profiling.py`
(`TraceCapture`, Perfetto capture over a sampling loop) and
`utils/ramtorch_helpers.py` (grad-flush / accumulator plumbing that differs
between resident and streamed stages, plus `prewarm_offload_staging` for
role-swapped LoRA under offload+cpu grad accum). **Model folders never import
from each other** — `chroma/` was built by copy-and-adapt from `krea2/`, not
by generalizing it, and `radiance/` from `chroma/` the same way; any future
model folder follows the same pattern.

## Architecture: dice once, flag the strategy

Since RamTorch 1.7 a `Pipeline` accepts a FLAT list of chunk modules and
decides independently how many GPUs to use and whether weights are resident or
streamed. So the model is diced exactly once — `krea2/model/chunks.py`,
one chunk per transformer block — and the hardware strategy is a config key
(`parallelism`) or CLI flag, not a separate code path:

| `parallelism` | GPUs | weights | when |
|---|---|---|---|
| `offload` | 1 | streamed from CPU RAM | one small / partially-free GPU |
| `pipeline` | N | resident on their stage | N GPUs, fits when split |
| `pipeline-offload` | N | streamed per stage | N GPUs, still doesn't fit |

Adding an execution mode means adding a flag, never a second trainer.

## Current state (2026-08-18)

`krea2/` contains:

- `model/chunks.py` — the dicing. `build_dit_chunks(dit, blocks_per_chunk=1)`
  -> `[embed, block x 28, head]`; `build_encoder_chunks(qwen, select_layers)`
  for the frozen Qwen3-VL; `balance_chunks_by_bytes` (exact DP — the embed
  chunk carries the whole text-fusion transformer, so an even split by COUNT
  leaves stage 0 heavy). DiT chunks relay
  `(combined, tvec, t_emb, freqs, attn_mask)`; `freqs`/`attn_mask` are built
  once in the embed chunk and shared instead of rebuilt per block.
- `train.py` — THE trainer. One `Pipeline` construction covers all three
  strategies x `mode: "lora" | "full"`. Loop is
  `pipe.step(...)` -> `flush_grads(1/k)` -> clip -> fused `AdamW` ->
  `zero_grads`. Frozen encoder runs as a forward-only pipeline on the same
  devices; the VAE is replicated per-GPU for chunk-parallel encode.
- `train_tdm.py` — TDM distillation (arXiv:2503.06674): distills the 28-step
  CFG teacher into a K-step (default 4) CFG-free student. The three networks
  TDM needs (teacher / fake score / student) share ONE frozen bf16 base and
  differ only by LoRA adapters switched per forward via
  `set_lora_role(dit, None | "fake" | "default")` — no extra weight copies.
  Data-free (captions only), same `parallelism` flag as `train.py`. The base
  must stay `requires_grad=True` under RamTorch (frozen by exclusion from the
  optimizers); see the 2026-08-18 worklog entry. Student checkpoints use the
  standard LoRA key convention, so `inference.py --lora-checkpoint ... --steps
  4 --guidance 0` runs them directly. NOTE: must run with `grad_ckpt: true` at
  any real batch — RamTorch stage workers ignore the driver's thread-local
  `no_grad`, so even `infer()` builds graphs (~5.3 GB vs ~0.5 GB per in-flight
  sample at 512px; see the 2026-08-18 worklog entry). First production run:
  tmux session `tdm`, `runs/k2-tdm-512/`, teacher = fullft step 49600,
  batch 12 x 8 microbatches at 512px (~140 s/iter on 4 GPUs).
- `train_utils.py` — K2 helpers: `vae_encode`, `vae_decode`,
  `_mu_from_seq_len`, `sample_timesteps`, `_pin_sdpa_backends`.
- `inference.py` — default (all resident on one GPU, the speed baseline),
  `--pipeline`, `--offload`, and the two combined. Offload has three weight
  tiers: `--offload-pin` (resident on GPU), CPU pinned RAM (default), and
  `--offload-nvme/--offload-nvme-path` (mmap'd from disk; **offload-only**,
  pipeline stages have no NVMe tier). Supports `--num-shards/--shard` for
  data-parallel fan-out of a prompt list, and
  `--profile/--profile-steps/--profile-warmup` for Perfetto traces.
- `tools/check_chunk_parity.py` — tiny model on CPU, chunked vs monolithic,
  forward AND every gradient, across 18 execution configurations. Seconds to
  run; all 18 bit-exact. Run it after any change to the dicing or relay.
- `tools/check_tdm_roles.py` — tiny model on CPU: each LoRA role matches a
  separately-built single-LoRA reference through the Pipeline, adapters are
  gradient-isolated, checkpoint keys match convention. Run it after any
  change to `model/lora.py` or the role plumbing.
- `model/` — mmdit, autoencoder, encoder, lora (multi-role: `extra_roles`,
  `set_lora_role`, `lora_role_keys`), sampling, chunks, plus `configs.py`
  (`MMDIT_CONFIGS`, `ENCODER_CONFIGS`).
- `configs/` — `train_{offload,pipeline,pipeline_offload}_{lora,full}.json`
  plus one `train_smoke.json` (any strategy via `--parallelism`/`--devices`),
  `train_tdm_{lora,smoke}.json` for the TDM trainer, and
  `train_mass_lora{,_smoke}.json` for the mass-LoRA trainer below.
  Dataset paths and `mmdit_checkpoint` are placeholders the user must point at
  real files (the smoke config points at a local e621 parquet on this machine).

### Mass LoRA: many adapters, one base (2026-08-24)

`krea2/train_mass_lora.py` trains L per-concept adapters SIMULTANEOUSLY off one
frozen base — no hotswapping and no extra weight copies. `model/lora_bank.py`
stacks every `nn.Linear`'s adapter into a bank (`[L, rank, in]`) and applies it
with a grouped `bmm` over a batch whose slots are contiguous on dim 0, so slot
i's gradient only ever touches adapter i and one base GEMM serves every slot.
Slots ROTATE (`slots_per_step` of L per step), which makes the per-microbatch
batch independent of L. The same `parallelism` flag applies; there is no `mode`
(a bank is always LoRA).

Three things differ from `train.py` and matter if you touch it:

- The shared norms / modulation tensors are **frozen by exclusion**. One copy
  serves every slot, so training them (as `train.py`'s LoRA mode does) would
  cross-contaminate the adapters.
- `utils/bank_optimizer.py::BankAdamW` replaces fused AdamW. Vanilla AdamW is
  incorrect under rotation — an inactive slot has zero grad but still drifts
  under momentum and weight decay — so it updates active rows only and keeps
  **per-slot** step counters for bias correction and warmup. Clipping is
  per-slot too.
- `dataloaders/mass_lora_dataloader.py` plans each step (one bucket, S slots,
  microbatch-major packing) from per-`(slot, bucket)` deficit targets. A slot
  with no samples in the chosen bucket is simply absent and keeps a zero
  gradient. Read the 2026-08-24 worklog entry before changing the planner: the
  obvious least-scheduled-slot heuristic starves whole buckets.

`tools/check_lora_bank.py` (18 CPU checks: bank == L solo models resident and
streamed, per-slot grad equivalence, exact zero for inactive slots, export
round-trip) and `tools/export_lora_bank.py` (slice slots into standard
`lora_A`/`lora_B` files that `inference.py --lora-checkpoint` consumes
unchanged) complete the folder.

### Tag embedding: booru tags as extra DiT tokens (2026-08-26)

`krea2/` can inject a learnable per-tag embedding. Matched tags become one
token each, projected to DiT width and concatenated onto the sequence in
`DiTEmbedChunk` between the text prefix and the image tokens. The point is to
combine SDXL-style bag-of-words tag prompting with natural language in one
model, so a user can write either or both.

It is **off unless a config carries a `tag_embed` block**, and everything is
routed through one class, `train_utils.TagTrainer`, which both `train.py` and
`train_mass_lora.py` use. Four properties the design rests on:

- **Permutation invariance is exact, not learned.** Every tag token gets the
  same RoPE position, and attention is permutation-equivariant, so tag order
  cannot change the output. Tag tokens are marked on RoPE **axis 0**, which
  text and image both leave at zero — a constant shared by all tag tokens, so
  the tag-to-tag relative rotation stays zero while tag-to-text and
  tag-to-image gets a usable "this is a tag" signal for free.
- **An untagged row is a no-op.** Masked slots are zeroed in the embedder and
  drop out of the attention mask, and torch's SDPA returns zeros (not NaN) for
  a fully-masked query row. `tags` is 100% present on the boorus and 0%
  everywhere else, so the non-booru sources supply the untagged case naturally.
- **Conditioning is decoupled.** Tags come from the parquet `tag_column`
  whichever caption column was sampled, so a natural-language caption arrives
  with the full tag set attached. That is the case that forces the table to
  carry information the text does not. `uncond_ratio` drops caption AND tags
  together (matching the CFG negative pass); `tag_drop_prob` drops tags alone.
- **`utils/row_optimizer.py::RowAdamW` is required, not an optimization.** A
  step touches ~500 of 315,966 rows; vanilla AdamW would apply weight decay and
  stale-momentum drift to the other 99.8% every step — the same bug
  `BankAdamW` solves for rotating slots. Moments (1.21 GB) park on the host.

`utils/tag_vocab.py` owns the vocabulary and matching. Matching runs two
passes over one normalizer (NFKC, casefold, `_`→space, unescape), so
`Long_Hair` / `LONG HAIR` / `long hair` are one id: an exact comma-segment
lookup (all the training column needs), then a word-level longest-match trie
for prose. The trie gates SINGLE-WORD tags on frequency, because the corpus
contains rare tags literally named `a` and `best` that would otherwise fire in
every sentence. Vocabularies are **versioned** — ids are positions in a file,
so a rebuild renumbers them; the row count is stored in the checkpoint and
asserted at load.

`tools/check_tag_embed.py` (28 CPU checks) covers chunked-vs-monolithic parity
with a live tag block, the bitwise no-op, permutation invariance, and the
matcher.

## Current state: chroma/ (2026-08-20)

Same shape as `krea2/`, targeting `lodestones/Chroma1-HD` (8.9B flux-style
DiT: 19 double + 38 single blocks, modulation distilled from an Approximator,
T5-XXL encoder + flux VAE, both auto-downloaded from the same HF repo; the
DiT base is a local safetensors file, `chroma_checkpoint`). Key differences
from krea2, all documented in the 2026-08-20 worklog entry:

- Dicing is 59 chunks (embed + 19 double + 38 single + head) relaying
  `(x, mod, pe, attn_mask)`; the modulation tensor rides the relay so its
  grads reach the Approximator in the embed chunk. `set_dit_seq` fixes the
  txt/img split before every step.
- The embed chunk (stage 0 = driver) is grad-checkpointed along with the
  blocks — the ported krea2 lopsided-driver-VRAM fix (there it was the
  TextFusion projector, here the Approximator + input projections).
- TDM in offload mode requires `prewarm_offload_staging` (see Key facts).
- `guidance` input to the model is always 0; CFG is applied externally in
  sampling (Chroma distilled guidance out of the model).
- Both trainers, inference, parity/role tools, and 9 configs mirror krea2's.
  Parity 18/18 bit-exact; TDM roles all pass; GPU smokes of both trainers
  exit 0 with sane losses and coherent previews.

## Current state: radiance/ (2026-08-23)

Radiance x0 patch-16: Chroma's transformer (19 double + 38 single, hidden 3072,
24 heads, `axes_dim [16,56,56]`, the same 5-layer Approximator producing 344 mod
rows, the same T5-XXL) with two swaps and **no autoencoder**:

- `img_in: Linear(64->3072)` becomes `img_in_patch: Conv2d(3, 3072, k=16, s=16)`,
  so the model consumes and emits `[B, 3, H, W]` pixels directly.
- `final_layer: LastLayer` becomes a NeRF decoder head: `nerf_image_embedder`
  (DCT positional embedding over each patch's raw pixels, the one **fp32** layer
  in the checkpoint) -> 4 x `NerfGLUBlock` (a hypernetwork whose
  `param_generator: Linear(3072->49152)` generates per-patch GLU weights) ->
  `nerf_final_layer_conv` (RMSNorm + `fold` + `Conv2d(64, 3, k=3)`).
- The head predicts **x0**; `v = (noisy - x0) / (t + eps)` converts it, so every
  v-space formula downstream (flow loss, Euler sampler, TDM's `x0 = x - t*v`) is
  unchanged. Base weights: `checkpoints/radiance/latest_x0.safetensors`, 659
  tensors, 9.506B params, strict-loaded (659/659, no missing/unexpected).

Patch 16 on pixels gives the **same token counts** as chroma's f8 VAE + patch 2
(`8*2 = 16`), so `_mu_from_seq_len`'s (256, 0.5) / (4096, 1.15) anchors,
`minres`/`maxres` and the dataloader's buckets all carry over untouched.
Wherever chroma writes `align = ae.compression * patch`, radiance writes
`align = patch_size`. The dataloader already hands back `[-1, 1]` images, which
is exactly the model's input space.

Differences that matter, all in the 2026-08-23 worklog entry:

- Dicing is **64 chunks** (embed + 19 double + 38 single + nerf-embed + 4 NeRF
  blocks + head). The transformer relays
  `(x, img_px, t, mod, pe, attn_mask)` with `out_no_grad=(1,2,3,4)`; the NeRF
  head relays `(img_dct, nerf_cond, img_px, t)`. `img_px` (the raw noisy image)
  must reach the far end twice — the NeRF embedder reads each patch's raw pixels
  and the head needs `noisy`/`t` for the x0->v residual, plus `H, W` for `fold`.
- Because the relay **changes shape** mid-list, no single `out_no_grad` index set
  describes every stage boundary. `utils/ramtorch_helpers.py` grew
  `set_resident_out_no_grad_per_stage`, which reads the declaration off each
  stage's LAST chunk instead.
- The **Approximator is frozen** (the reference's choice): it runs under
  `torch.no_grad()` in the embed chunk and is excluded from LoRA and from both
  optimizers, so `mod` is a pure no-grad relay and 62 chunks stop computing
  `dL/dmod`. `chunk_params_by_stage`/`build_offload_adamw` grew `exclude_ids`
  for this (you cannot `requires_grad_(False)` under RamTorch).
- `x0_eps` is **explicit state** (`set_x0_eps`), not `self.training`: `train.py`
  uses `5e-2` to match its target `(noisy - x1) / (t + 5e-2)`; `train_tdm.py`
  and inference use `0.0` because they invert the residual.
- `LoRA excludes `nerf_image_embedder`` (`Linear(67->64)` — a rank-32 adapter is
  bigger than the layer, the same pathology as krea2's TextFusion projector).
- **`txt_pos_ids: "zeros"`**, settled empirically, NOT as the plan assumed. See
  the worklog; `radiance/tools/check_txt_pos_ids.py` is the tool that settled it.
- TDM's `pack_targets` concatenates on the **channel** dim
  (`[x0_tgt(3) | x_in(3) | t(1) | w(1)]` -> `[B, 8, H, W]`) since the output is
  4-D; noise is sampled at pixel resolution; previews need no decode.

Verified: parity 18/18 bit-exact across all execution modes, TDM roles all pass,
strict 659/659 load, and 6-step smokes of both trainers in **both** offload
(peak 5.72 / 5.33 GB) and 4-GPU resident pipeline (peak 7.0-8.5 GB/GPU) exit 0
with matching losses and coherent previews.

## Key facts

- Launch is **plain `python`** (one process drives all pipeline stages via
  threads) — NOT torchrun/DDP.
- Effective batch = `batch_size` (per microbatch) x `n_microbatches` in every
  strategy; microbatches serve as gradient accumulation.
- `mode` (lora/full) and `parallelism` (offload/pipeline/pipeline-offload) are
  independent axes — the two words are easy to confuse in configs.
- K2 DiT base weights are a local safetensors file (`mmdit_checkpoint`);
  the VAE (`Qwen/Qwen-Image`) and encoder (`Qwen/Qwen3-VL-4B-Instruct`)
  auto-download from HuggingFace.
- SDPA backends must be pinned cudnn-first once per process
  (`_pin_sdpa_backends` in `krea2/train_utils.py`) — pipeline worker threads
  racing on `sdpa_kernel()`'s global flags otherwise break kernels, and the
  math backend would blow up memory on the DiT's bool-masked attention.
- For RESIDENT stages the trainer aliases `p.grad` to the stage's grad
  accumulator instead of `result.flush_grads()`, avoiding a second full fp32
  grad copy (~14 GB/GPU). STREAMED stages use the engine's `flush_grads()`
  (its buffers are persistent and reused). `utils/ramtorch_helpers.py` owns
  that split.
- Gradient accumulation under streaming (`grad_accum`, 1.8) is chosen by
  `mode`, NOT by the parallelism: `"stream"` (accumulate on the GPU, cross PCIe
  once at flush) is 26% faster for full FT but 22% slower for LoRA, where the
  trainable slice is small enough that the D2H-plus-host-add path wins and the
  GPU accumulator slots would only thrash. Numbers in the worklog.
- Offload full FT is **host-bound, not PCIe-bound**: ~35 s per step at
  256px/batch 4 vs ~4.4 s for LoRA, with `acquire_wait_s` ~0 in both. The
  trainer's `Time split:` line attributes it — grad flush 34%, grad clip 13%,
  CPU AdamW 14%, all of it passes over 51 GB of fp32 masters/grads. The same
  step on RESIDENT stages across 4 GPUs takes 6.4 s and spends ~0% there, so
  stream weights for headroom (18.6 vs 68.7 GB peak), never for speed.
- Offload full FT needs ~256 GB host RAM (fp32 masters + grad accumulators +
  pinned flush buffers + AdamW state for 12.8B params).
- Mass LoRA's cost scales with ACTIVE slots per step, never with the number of
  adapters L — that is the whole point of the bank. The exception is the CPU
  optimizer under `offload`: ~3 s/step for 8 active rank-16 slots on
  `large_wide`, memory-bandwidth bound on the fp32 moments (2% of the step on
  resident GPU stages).

## Roadmap

- [x] Second model folder following the same per-model shape — done as
      `chroma/` (2026-08-20): copied `chunks.py` + trainers and re-diced,
      the execution strategies came for free.
- [x] Third model folder — done as `radiance/` (2026-08-23), and the first one
      with no VAE and a non-uniform relay. Confirms the pattern scales: the
      only shared-infra change needed was `set_resident_out_no_grad_per_stage`
      plus `exclude_ids` on the optimizer builders.
- [x] Mass LoRA for krea2 (2026-08-24): L adapters trained at once off one
      frozen base via stacked banks + grouped bmm. Fourth execution shape for
      the same chunk list, no dicing or relay change.
- [ ] Production mass-LoRA run (verified only to 6-step 256px smokes on a
      3-slot `rating` grouping). Real runs want a real `group_column` (artist /
      character) and should watch the trainer's bank-budget print.
- [ ] Port the bank to `chroma/` and `radiance/` once mass LoRA is proven in
      production (copy-and-adapt, per the per-model convention).
- [ ] Production radiance run (the port is verified only to 6-step smokes at
      256px). Watch the LAST pipeline stage: the byte-balancer cannot see the
      NeRF head's `[B*N, 49152]` generated-weight activations, so a manual
      `chunks_per_stage` may be needed at 1024px.
- [ ] Call `prewarm_offload_staging` from `krea2/train_tdm.py` too (same
      latent offload+cpu-accum+role-swap KeyError; krea2 TDM has only ever
      run in pipeline mode).
- [ ] Scripted offload-vs-baseline inference benchmark (currently manual: run
      `krea2/inference.py` with and without `--offload` on the same
      prompt/seed).
- [ ] Reduce offload full-FT optimizer cost (e.g. 8-bit or sharded CPU
      optimizer state) — currently the dominant per-step cost.

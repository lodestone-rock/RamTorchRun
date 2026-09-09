# Krea-2 Turbo attention search: grid and orthogonal-array proposal

## Status and scope

The initial experiment was **three coverage controls, not an orthogonal array**.
User subsequently approved running 160 images without waiting for calibration.
The completed L8 run uses **spread-offset levels**, described below. All160
images verified successfully. The active extension is the four-level L64.
The later U/C sensitivity design remains a separate, unlaunched proposal.
Per-layer selection is implemented; importance probing is not. No policy here
should be described as calibrated.

## Active L64: four query offsets per band

Artifacts: `/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-turbo-head-probe/l64_spread_offset/`.
Same runner and recipe as L8; no attention-model code changed for this extension.

- OA(64,7,4,2): seven four-layer bands; levels0/1/2/3 keep heads
  `4*g + level` full for g=0..11. Other36 heads use local11, at all8 steps.
- Enumerate a,b,c in 0..3 and use columns
  `[a,b,c,a XOR b,a XOR c,b XOR c,a XOR b XOR c]`. These are linear forms
  over GF(4), with XOR as field addition, NOT addition modulo4. Every column
  contains each level16 times; every pair contains each of16 level pairs4 times.
- All eight original L8 rows occur exactly: L8 rows0..7 map to L64 rows
  0,1,4,5,16,17,20,21. The launcher verified checkpoint identity metadata,
  exact prompts/seeds, per-layer masks, sampling recipe and all PNGs before
  copying these rows. `reuse.json` preserves each source; originals unchanged.
- 64x20 = 1,280 final images: 160 reused + **1,120 new renders** (56 rows).
  Pending rows are assigned round-robin to GPUs0/1/2/3, 14 each. Frozen queues
  are in `schedule.json`; resume requires the same GPU list.
- `sweep.log` reports progress; per-GPU status JSON updates atomically after
  each row; final `status.json` includes reused rows as well as new results.
  Reused rows retain their original command provenance, not a fictitious new run.
- This covers all query offsets but NOT independent choices per KV group or
  per layer. It is strength-two screening, not the exhaustive 4^7 factorial;
  interactions can remain aliased. No automatic quality ranking is claimed.

```bash
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/sweep_attention.py \
  --levels 4 --prepare-only \
  --reuse-from /mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-turbo-head-probe/l8_spread_offset
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/sweep_attention.py \
  --levels 4 --devices 0 1 2 3
```

CPU validation: `/mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_attention_sweep.py`
checks the array, L8 inclusion, budgets, idempotent reuse and mismatch rejection.

## Completed L8: spread-offset screening (2026-09-06)

Runner: `/mnt/datapool_u2/lodestone/RamTorchRun/krea2/sweep_attention.py`.
Artifacts: `/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-turbo-head-probe/l8_spread_offset/`.

- Uses the same L8 table and seven four-layer bands shown in Stage2 below, but
  **NOT its U/C levels**. Every band has level0 = heads 0,4,8,...,44, and
  level1 = heads 1,5,9,...,45. Each level keeps one query per KV group full.
- Both levels have exactly 12 full + 36 local11 heads per layer, all eight steps.
  Level1 is an arbitrary within-group offset, not an importance-selected policy.
  This search covers only offsets0/1, not all four queries in every KV group.
- All eight rows are rendered anew: 160 images. Row0 repeats the spread control;
  use it to assess reproducibility, not as an independent new head policy.
- Four independent GPU workers: GPU0 rows0/4, GPU1 rows1/5, GPU2 rows2/6,
  GPU3 rows3/7. Each loads one row at a time with RamTorch offload. GPU selection
  is pinned with CUDA_VISIBLE_DEVICES; each child uses logical cuda:0.
- `design.json`, `policy_00.json` through `policy_07.json`, and `prompts.json`
  freeze the definition. Manifests record every layer's actual head list.
- `sweep.log` reports launches/verification/failures; `row_00.log` etc. contain
  inference logs; `row_00/` etc. contain PNGs/manifests. `status.json` summarizes
  all rows at completion; per-GPU status files are written as each worker ends.
- Each row verifies all20 images, prompt/seed pairs, checkpoint recipe, sampling
  settings and exact per-layer masks before reporting success. Failures are
  recorded; other rows continue. No silent retries. A single-process lock stops
  duplicate launchers for this output root. Partial rows rerender on manual
  restart; completed rows are verified and skipped. Use the same GPU list when
  restarting (command provenance is frozen).
- Same fixed recipe and aliasing limitations as below. This exploratory run
  intentionally bypasses the earlier calibration decision gate at user request.

```bash
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/sweep_attention.py --prepare-only
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/sweep_attention.py --devices 0 1 2 3
```

## Fixed experiment contract

| Setting | Value |
|---|---|
| Backbone | `/mnt/datapool_u2/lodestone/RamTorchRun/checkpoints/krea2/fullft_step_49600.safetensors` |
| Adapter | `/mnt/datapool_u2/lodestone/RamTorchRun/checkpoints/krea2/turbo_delta_r512_fullft49600.safetensors` |
| Adapter application | Live rank512 / alpha512 / scale1.0, including norm/bias overrides; no artist adapter |
| Sampling | 1024x1024, 8 Euler steps, guidance0, mu1.15, batch1 |
| Prompt/seed pairs | `/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-turbo-head-probe/nonexplicit20.json` |
| Source indices | 2,3,8,9,10,12,13,17,22,30,31,69,73,74,80,83,88,89,95,97 |
| Execution | Single-GPU RamTorch offload, window2, pin4; independent cases may run on different GPUs |
| Local image neighborhood | Centered, clipped 11x11 in image-token coordinates; no row wrap |
| Prefix | Valid prefix queries global; image queries see every valid prefix key |
| Unchanged modules | Text fusion, text encoder, QKV/gates/output projections, RoPE, VAE |

Original prompts and nonconsecutive seeds are preserved. Historical preview PNGs
are not matched controls for this recipe; the regenerated dense set is the
reference. Filenames use subset indices; `source_metadata` preserves original
manifest indices. GPU identity is a scheduling detail, not a search factor;
timings under contention are not comparable speed benchmarks.

## Underlying mask tensor

Define `F[t, l, h]` as 1 for full attention and 0 for prefix-plus-local attention:

```text
F: [8 denoising steps, 28 joint DiT layers, 48 query heads]
                                               |
                                      12 KV groups x 4 queries

h = 4*g + r,  g in [0,11], r in [0,3]
```

GQA shares K/V within a group, not queries or head importance. Query-head IDs
do not imply the same role across layers. At the sparse budget:

```text
sum_h F[t,l,h] = 12   for every t,l
36 remaining query heads use 11x11
```

There are 10,752 binary cells without constraints. A timestep-static policy has
1,344 cells. Independently selecting 12/48 at each layer has
`binomial(48,12)^28` possibilities; additionally enforcing one per KV group
reduces this to `4^(12*28)`. Both are still too large for exhaustive rendering.
The initial implementation broadcast ONE head selection over all layers and
steps. The current CLI also accepts timestep-static per-layer lists through
`--local-layer-heads`; step-dependent policies remain conceptual.

## Stage 0: actual control grid

| Case | Full query heads in each layer | Local heads | Step schedule | Initial assignment |
|---|---|---:|---|---|
| `dense` | All 48, normal dense backend | 0 | Dense all 8 | GPU2, completed and verified |
| `local12_first` | 0 through 11 | 36 | Same mask all 8 | GPU2 |
| `local12_spread` | 0,4,8,12,16,20,24,28,32,36,40,44 | 36 | Same mask all 8 | GPU3 |

Shape: **3 policies x 20 fixed prompt/seed pairs = 60 images**. Only the two
sparse rows share the 12-head compute budget. First12 covers three KV groups;
spread12 covers all twelve. This tests coverage, not individual head importance.
The same sparse head count/layout gives the same block counts across these two
policies for a given input, although numerical outputs differ.

Artifacts are under
`/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-turbo-head-probe/`.
The original `run_sweep.sh` was launched sequentially, then its parent was stopped
without stopping its first12 worker. `run_spread_gpu3.sh` runs spread12 separately.
Use `first_verify.log` and `spread_driver.log` for completion verification; the
original `sweep.log` no longer monitors both cases. Do not relaunch these scripts
against in-progress output folders.

### Stage 0 result and decision gate

All three cases completed; each passed image/seed/recipe verification (20 each).
User visual assessment: first12 and spread12 have similar quality and duplication;
spread12 has slightly fewer duplicates, but no substantial perceived improvement.
This is qualitative feedback, not a measured statistical-significance result.

KV-group coverage alone did not resolve the observed failure. This does NOT rule
out calibrated layer-specific head selection: neither control selected heads by
sensitivity. Treat both as weak controls, not evidence that all head subsets are
equivalent. Do not expand into more arbitrary global head permutations yet.

Before spending 160 renders on L8, collect Stage1 scores and test the all-U and
all-C calibrated policies on a small development subset. If both still fail to
improve composition, reconsider the factor choice rather than automatically
running L8. A separate 1-2 dense-initial-step test can probe whether preserving
early global processing helps; it changes computation and is not a matched-budget
head-selection comparison. No new runs have been launched based on this feedback.

## Stage 1: reduce head choices by sensitivity (proposed)

On dense Turbo trajectories, evaluate dense and 11x11 attention from the same
Q/K/V at each layer and step; propagate only dense outputs. Accumulate per-head
mean squared output differences **after the sigmoid gate**, over valid image
queries. Optionally refine shortlisted candidates with their output-projection
slices. Keep step-resolved scores before aggregating. No full attention matrices
need to be retained. This measures local perturbations, not end-to-end quality.

Build two fixed candidate sets independently for each layer:

- **U (unconstrained):** 12 highest-scoring query heads out of 48.
- **C (coverage-constrained):** highest-scoring query head within each of 12 KV
  groups. Exactly 12 full heads, but not necessarily the same heads as U.

Freeze scores, candidate lists, aggregation rule and tie-breaking before the next
stage. Suggested tie-break: ascending head ID. For the initial design, aggregate
all eight steps equally and keep selections timestep-static. If U equals C in a
layer group, that factor has no actual effect; record the degeneracy rather than
claiming seven distinct effects.

## Stage 2: concrete L8 orthogonal grid (proposed)

Group the 28 layers into seven consecutive four-layer bands. These are search
compression groups, not proven architectural boundaries or denoising phases.

| Factor | Zero-based layers | Level 0 | Level 1 |
|---|---|---|---|
| A | 0-3 | U for each layer | C for each layer |
| B | 4-7 | U for each layer | C for each layer |
| C | 8-11 | U for each layer | C for each layer |
| D | 12-15 | U for each layer | C for each layer |
| E | 16-19 | U for each layer | C for each layer |
| F | 20-23 | U for each layer | C for each layer |
| G | 24-27 | U for each layer | C for each layer |

Here factor C is a column label; candidate C means coverage-constrained.
Each factor changes only head identity, not head count, window or timestep
schedule. Different layers within a band still have their own candidate lists.

The complete two-level factorial has `2^7 = 128` policies. Use
**OA(8,7,2,2)** (L8): eight runs, seven factors, two levels, strength two.

| Run | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 | 0 | 1 | 1 | 1 |
| 2 | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| 3 | 0 | 1 | 1 | 1 | 1 | 0 | 0 |
| 4 | 1 | 0 | 0 | 1 | 1 | 0 | 1 |
| 5 | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| 6 | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| 7 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |

Construction: enumerate `(a,b,c)` in binary order, then columns
`[a,b,c,a XOR b,a XOR c,b XOR c,a XOR b XOR c]`. Each level appears four times
per column; every level pair appears twice for every pair of columns.

Shape if all reference pairs are used: **8 policies x 20 pairs = 160 images**,
versus 2,560 for the full factorial. Dense controls are reused. All-U is row0;
all-C is NOT a row and should be rendered as a separate confirmation if useful.
Four GPUs can execute four independent policy rows per wave (two waves), subject
to memory checks. Each row still uses single-GPU offloading, not pipeline mode.

### Interpretation and confirmation

This is a saturated main-effect screening design, **not** an interaction-resolving
or optimal-policy search. For example, D is A XOR B, so its main effect is aliased
with the A:B interaction. Diffusion errors can interact across layers and steps.
Prompt repetitions improve coverage but do not remove this aliasing.

Compare each policy to its matched dense output. Record global composition,
unwanted subject duplication, prompt adherence and detail separately. Some
prompts explicitly request grids, repeated views or multiple subjects: count
only unwanted repetition. Image/denoiser error metrics are supporting diagnostics,
not substitutes for composition assessment. Define any scalar ranking rubric
before examining the L8 outcomes; no automatic quality metric is implemented.

Use mean paired differences between factor levels for screening, then explicitly
render the assembled best-level policy and check it against the best measured
row. Test important interactions using follow-up budget-preserving swaps or a
foldover design; L8 alone does not justify causal claims about individual heads.

If the same 20 prompts inform scores, policy selection or visual iteration, they
are development data, not held-out evaluation. Reserve a split BEFORE calibration
or obtain new non-explicit prompt/seed pairs for final confirmation. The
160-image count above assumes using all20 for development, not a claimed holdout.

## Separate follow-up axes (not crossed into L8 yet)

- Dense-start schedule: 0, 1 or 2 initial high-noise steps dense, selected sparse
  policy afterward. This changes the budget: mean full-head fraction is
  `(48*d + 12*(8-d))/(48*8)` for d dense steps: 25%, 34.375%, 43.75%.
  This fraction is not active-block occupancy or a predicted speedup.
- Head budget: 12 versus larger budgets only after evaluating identity at fixed12.
- Layer allocation: move full-head budget between layers only as a separate test;
  raw discrepancy scores across layers are not necessarily directly comparable.
- Window size stays 11x11 for this search. Changing it is a separate experiment.

Still needed before calibrated Stage1/2: gated importance accumulation and a
quality report. Per-layer policies, preparation/caching, serialization and the
spread-offset OA runner are now implemented. CLI accepts one global list via
`--local-full-head-ids` OR a JSON list of 28 lists via `--local-layer-heads`.
Training and pipeline execution remain unsupported for this experiment.

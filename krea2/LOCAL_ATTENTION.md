# Uncalibrated local-attention inference

Search axes, current controls and the proposed L8 orthogonal array are documented
in `/mnt/datapool_u2/lodestone/RamTorchRun/krea2/ATTENTION_SEARCH.md`.

Opt-in experiment; dense attention remains the default. No checkpoint conversion,
fine-tuning or calibration is performed.

```bash
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/inference.py \
  --config /mnt/datapool_u2/lodestone/RamTorchRun/krea2/configs/train_pipeline_lora.json \
  --mmdit-checkpoint /mnt/datapool_u2/lodestone/RamTorchRun/checkpoints/krea2/raw.safetensors \
  --no-lora --offload --offload-window 2 --device cuda:2 --batch-size 1 \
  --local-attention --local-full-heads 2 --local-window 11 \
  --width 1024 --height 1024 --steps 28 --guidance 4.5 --seed 42 \
  --prompt 'A red fox sitting in a snowy pine forest, detailed wildlife photograph, soft morning sunlight.' \
  --out-dir /mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-local-attention-ab/local
```

Remove `--local-attention` for the dense control; use a different output folder.
Keep checkpoint, prompt order, seed, batch size, resolution, steps and guidance
identical. The manifest records the attention policy and base checkpoint.

## Exact policy

- Every one of the 28 joint DiT blocks uses the same rule.
- **Query heads 0 and 1 of 48** are full. The other 46 use an 11x11 image-token
  neighborhood. This is intentionally much more aggressive than 2/12 full.
  `--local-full-heads 8` tests the 2/12 fraction; 48 tests the dense-mask limit.
- Valid text/tag queries attend globally. Image queries see all valid text/tag
  keys plus their clipped 2D neighborhood. No row wraparound.
- Padding/invalid text and tag slots remain masked for queries and keys.
- Image coordinates come from `prepare` positions, so rectangular grids work.
- Text fusion, the Qwen text encoder, QKV projections/gates, RoPE and VAE are
  unchanged. Native GQA remains 48 query heads / 12 KV heads.

Supported: single-GPU resident or RamTorch `--offload`. Pipeline mode is rejected:
the shared per-forward policy is deliberately not a thread-safe per-stage cache.
Training with this policy raises an error. Checkpoint keys and chunk tuple shape
are unchanged. A small LRU caches masks separately for CFG validity/layouts; it
still performs small device-to-host cache-key copies per forward. This is a
quality probe, not an optimized implementation or speed benchmark.

Validation:
`uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_local_attention.py`
checks GQA48/12 vs masked SDPA, invalid prefix/tag slots, padding, rectangle
boundaries, cache reuse, the all-full limit and tiny-model monolithic/chunked/
RamTorch-offloaded output parity. Run the existing chunk parity tool as well.

## Initial A/B result (2026-09-06)

Completed two matched base-checkpoint samples with RamTorch offload on GPU 2:
1024x1024, 28 steps, guidance 4.5, batch size 1, seeds 42/43. Outputs and
manifests are in `/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-local-attention-ab/`
under `dense/` and `local/`. Both processes completed successfully.

- Fox: the dense control has a coherent single subject; local attention repeats
  foxes across the scene and shifts toward an illustrated appearance.
- Bookstore: local attention repeats windows and bicycles across an implausible
  facade. The dense control itself has an unusual small storefront surrounded
  by black, so it is not a clean quality baseline.
- This is severe global-composition degradation in a two-prompt smoke test, not
  a quality benchmark. Local detail and prompt motifs survive. No calibration
  or head selection was performed: heads 0/1 are arbitrary and share a KV group.
- The logged local-mask occupancy includes a 29.15% active-block layout with
  512 prefix slots and 4096 image tokens; validity differs between CFG inputs.
  Occupancy is relative to the padded dense block grid, not a speedup claim.
- Numerical checks passed (maximum GQA reference error 1.55e-6), including the
  all-full limit and RamTorch offload parity. Existing chunk parity: 18/18 pass.
  Timings are not representative on the contended machine.

## Second A/B result (2026-09-06): 12/48 full heads

Matched run with `--local-full-heads 12` (1/4 of query heads), everything else
identical. Outputs under `runs/k2-local-attention-ab/local12/`. Logged mask
occupancy: 42.21% / 37.04% active blocks (CFG cond/uncond layouts; 512 prefix,
4096 image tokens, padded 4608).

- Fox: **still 7 repeated foxes** (dense control: 1). More and sharper than the
  2-head run, but global composition is not restored.
- Bookstore: **4 bicycles** and repeated glowing windows across the facade
  (dense control: 1 bicycle, single storefront). Same failure mode as 2 heads,
  visually richer.
- Conclusion: going from 2/48 to 12/48 uncalibrated full heads does not rescue
  global composition. The failure is not rescued by a naive head-count
  increase; the heads chosen (first N, arbitrary, no calibration) likely lack
  the long-range/global roles, or the model relies on most heads having global
  receptive fields. Any usable version of this needs head calibration/selection
  or fine-tuning, not just a bigger N.

## Turbo manifest sweep (2026-09-06)

Run artifacts and the fail-fast sequential launcher live at
`/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-turbo-head-probe/`.
`nonexplicit20.json` selects source indices
2,3,8,9,10,12,13,17,22,30,31,69,73,74,80,83,88,89,95,97 from the historical
99-prompt manifest, with original prompts and nonconsecutive seeds unchanged.
The historical images are NOT matched controls: this run regenerates dense.

- Fullft step49600 + turbo delta r512, alpha512, scale1.0 (live LoRA, including
  checkpoint norm/bias overrides), no artist adapters. All cases use the same
  loading path; this is not the exact full-rank Turbo merge.
- 1024px, 8 Euler steps, guidance0, mu1.15, batch1, RamTorch offload.
  Dense completed on GPU2; first12 and spread12 were assigned GPU2/GPU3 in
  parallel after stopping only the original sequential launcher parent.
- Cases: `dense`, `local12_first`, `local12_spread` (all local windows11).
- Explicit selection is now supported via `--local-full-head-ids` followed by
  N unique query-head indices; count must match `--local-full-heads`.
  Spread selection: 0 4 8 12 16 20 24 28 32 36 40 44 (one per KV group).
- Manifest-provided seeds now drive actual noise, not just metadata. Currently
  requires batch1; seedless input retains the existing seed+index behavior.
  Input metadata is nested under `source_metadata` so old filenames, attention
  settings and checkpoint fields cannot overwrite actual output provenance.
- GPU GQA reference checks include spread heads; tiny offload parity and 18/18
  chunk checks passed. Sweep verifies every image and seed before advancing.
- This is a coverage-control sweep, not yet a calibrated head-importance search.
- All three cases completed and passed verification (20 images each). User
  visual assessment: similar quality and duplication for first12/spread12;
  spread12 has slightly fewer duplicates without a substantial perceived gain.
  This is qualitative, not a statistical test. Coverage alone did not fix the
  failure; calibrated per-layer head selection remains untested.

## L8 spread-offset extension

User approved 160 new renders without calibration. See the active-run section of
`/mnt/datapool_u2/lodestone/RamTorchRun/krea2/ATTENTION_SEARCH.md` for exact levels,
grid, scheduling and artifact paths. `--local-layer-heads <json>` accepts one
head-ID list per joint layer (28 lists for large_wide), mutually exclusive with
the global head-ID flag. Each list must have `--local-full-heads` unique IDs.
Only distinct policies build/cache masks; blocks hold explicit policy references
rather than selecting via a mutable execution counter. Embed prepares every
distinct mask; no chunk tuple or checkpoint keys changed. Output manifests store
`attention.layer_head_ids` and leave global `full_head_ids` null for these runs.
GPU tests now check offset0/1 GQA masks, per-layer SDPA agreement and tiny-model
chunk/offload parity; 18/18 existing chunk configurations still pass.

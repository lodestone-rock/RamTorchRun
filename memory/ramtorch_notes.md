# RamTorch notes (as used in this repo)

RamTorch is installed from PyPI (`ramtorch>=1.8.0`). The library source is
symlinked at `RamTorch/` for reading only — the deep references are:

- `RamTorch/docs/pipeline_parallel.md` — full pipeline guide
- `RamTorch/docs/offload.md` — full OffloadModel guide
- `RamTorch/examples/` — many small runnable demos (MNIST pipeline/offload,
  schedule benchmarks, correctness checks)

## The core idea in this repo: dice once, flag the strategy

Since 1.7 a `Pipeline` can be built from a FLAT list of chunk modules and
decide independently how many GPUs to use and whether weights are resident or
streamed. So `krea2/model/chunks.py` dices the model once (one chunk per
block) and all three strategies come from one construction:

```python
Pipeline(chunk_modules=chunks, chunks_per_stage=counts, devices=devs,
         autocast=torch.bfloat16, offload=..., offload_window=..., ...)
```

| `devices` | `offload` | strategy |
|---|---|---|
| 1 | True | single-GPU weight streaming |
| N | False | resident pipeline parallel |
| N | True | pipeline parallel + per-stage streaming |

`OffloadModel(chunks, ...)` is the same engine used directly, and is still the
path `inference.py --offload` takes (it is the only one with an NVMe tier).

### The chunk contract

A chunk returns a tensor or a tuple whose elements become the next chunk's
positional args. Rules that actually bite:

- RamTorch flags every FLOAT chunk input as a grad-requiring leaf. A
  parameter-free float passthrough (our RoPE table) must be `.detach()`ed
  inside the consumer, or every chunk pays a pointless `dL/dfreqs`.
- At a stage boundary, `needs_grad[i] = (i not in out_no_grad) and
  output[i].is_floating_point()`. Streamed stages read `out_no_grad` off the
  last chunk module; RESIDENT stages are wrapped in the private
  `_ChunkSequential`, which has none — set it after construction
  (`utils/ramtorch_helpers.set_resident_out_no_grad`).
- Bool tensors are never grad-flagged, so masks relay for free.
- Anything a later chunk needs must be relayed through every chunk in between
  (our `t_emb` rides 28 blocks to reach the head).

`krea2/tools/check_chunk_parity.py` checks all of this on CPU in seconds:
tiny model, chunked vs monolithic, forward AND every gradient, across 18
configurations (resident/streamed x 1/2 stages x keep/checkpoint x window
1/2 x activation offload x grad_accum modes). All 18 are currently bit-exact.

## APIs this repo uses

### `ramtorch.Pipeline`

Used by `krea2/train.py` (all strategies) and `inference.py --pipeline`.

- Training: `pipe.step(nested_microbatch_inputs, targets=..., schedule="staggered_1b1f", n_microbatches=N, loss_fn=...)`
  returns a result with `.loss`; grads land in per-stage accumulators.
- Inference: `pipe.infer(inputs, n_microbatches=N)` — forward only.
- `chunks_per_stage` controls the split. Balance by WEIGHT BYTES, not chunk
  count: our embed chunk carries the whole text-fusion transformer (~1B) vs
  ~0.4B for a block (`chunks.balance_chunks_by_bytes`, exact DP).
- Grad flush differs by stage type, hence `utils/ramtorch_helpers.flush_grads`:
  streamed stages use `st.flush_grads(scale)` (engine writes `.grad` on the
  CPU masters, then invalidates GPU residency); resident stages alias
  `p.grad` to `st.grad_acc[p]` and scale in place, avoiding the second full
  grad copy `result.flush_grads()` would allocate (~14 GB/GPU in full FT).
- `st.zero_grad_acc()` works for both.

### `ramtorch.OffloadModel` / `OffloadStage` (CPU -> GPU weight streaming)

- Masters live in CPU pinned memory; a `window` of chunks streams through the
  GPU (>= 2 overlaps H2D with compute); `pin` keeps N chunks resident. Peak
  weight VRAM ~ `(window + pin) / chunks` of the model.
- **Inference**: call like a module, `out = model((img, txt, t, pos, mask))`.
  `forward()` is `@torch.no_grad()`.
- **Training** (this is the whole API, `loss.backward()` is NOT the path):

  ```python
  res = model.step(x, targets=y, loss_fn=mse)   # k times to accumulate
  model.flush_grads(scale=1 / k)                # sets .grad on the CPU masters
  torch.nn.utils.clip_grad_norm_(trainable, 1.0)
  opt.step()                                    # AdamW(fused=True) -> fused CPU kernel
  model.zero_grad_acc()
  ```

- `keep_activations`: `False` = recompute each chunk's forward (cheapest VRAM,
  dropout resamples), `True` = keep every chunk's graph (fastest), or
  `"checkpoint"` = per-chunk torch checkpoint (recompute-level memory,
  dropout-safe). Exposed as `offload_backward`, default `"checkpoint"`.
  **`False` is rejected by `OffloadStage`** — its no-grad forward leaves the
  pipeline's loss graph disconnected — so the trainer only offers the other
  two.
- `grad_accum="stream"` (1.8 default) accumulates gradients ON the GPU and
  spills once at flush; `"cpu"` is the D2H-per-microbatch + host-add path.
  **Pick it by how much of the model is trainable, not by taste**: measured on
  K2, stream is 26% faster for full FT (30.0 vs 40.7 s/step) and 22% SLOWER for
  LoRA (7.5 vs 5.8). With a small trainable slice the host-side adds are cheap
  and the GPU accumulator slots just thrash — watch `acc_evictions` in `stats`.
- `offload_activations=True` streams saved activations to pinned CPU. Pairs
  with `keep_activations=True`; under `"checkpoint"` the engine already caches
  only chunk boundaries, so it buys little. Measured on K2 LoRA at 256px: keep
  backward 26.3 -> 10.9 GB peak for +12% step time. On full FT it is not the
  lever at all — weights + grads + Adam state are 51 GB/GPU of a 68 GB peak.
- `pin` is clamped to the stage's chunk count, so **`offload_pin` >= that count
  = resident weights through the offload engine** (`loads: 0`, masters stay on
  the GPU, fused CUDA AdamW). The one reason to use it: `offload_activations`
  only exists on the OffloadStage path.
- **Bug (1.8.0)**: all chunks pinned + `keep_activations=True` +
  `offload_activations=False` faults with `CUDA error: unspecified launch
  failure` in `_grads_for` during backward — deterministically at step 5 of the
  K2 full-FT bench, reproduced twice, after 5 clean steps. `checkpoint`, or the
  same config with activation offload ON, both run fine.
- `OffloadAdamW` (private `ramtorch.offload_optimizer`) streams optimizer state
  through a GPU window. Sharded one-per-stage and stepped in threads it still
  lost to fused CPU AdamW on K2 full FT (41.3 vs 30.5 s/step) — its docstring
  is right that one pass over DDR beats 28 B/param over PCIe. Reach for it only
  when the state does not fit in host RAM.
- Finish with `close()`. `stats` = `{"loads", "nvme_loads", "acquire_wait_s"}`
  plus `acc_loads`/`acc_evictions` (stream accumulators) and
  `act_offloads`/`act_reloads`/`act_bytes_offloaded` (activation offload).
- **Profiling**: `step(profile_path=...)` traces the training path; `forward()`
  has NO hook, so for inference we drive `torch.profiler` ourselves and splice
  in the worker-thread spans (`utils/profiling.py::TraceCapture`). The recipe
  is `off._span_log = []` -> `record_function("offload_clock_sync")` paired
  with `time.monotonic_ns()` -> `OffloadModel._inject_thread_spans(path, spans,
  sync_us)`. Kineto only records `record_function` on the thread that entered
  the profiler, so the H2D loader track is otherwise missing entirely. With
  several engines (one per streamed stage) the track names must be made unique
  BEFORE injection — the injector allocates trace thread ids per track name.

## Gotchas

- **SDPA backend pinning**: pin `[CUDNN, FLASH, EFFICIENT, MATH]` once per
  process with `set_priority=True`, keep the context object referenced forever,
  and disable k2.mmdit's per-call `sdpa_kernel` context
  (`krea2.model.mmdit.set_sdpa_ctx(False)`). See `_pin_sdpa_backends` in
  `krea2/train_utils.py`. Pipeline worker threads racing on the process-global
  SDPA flags otherwise corrupt the backend state, and the math backend OOMs on
  the DiT's bool-masked attention. (On CPU, `set_sdpa_ctx(False)` alone —
  CUDNN_ATTENTION has no CPU backend, which is why the parity harness sets it.)
- One process, plain `python` — never torchrun for these scripts.
- The first pipeline device ("driver") also hosts noise/VAE/decode tensors —
  give it the freest GPU.
- `zero_grad_acc()` (in-place zero), never `opt.zero_grad(set_to_none=True)`,
  because `p.grad` is aliased to the stage accumulator (resident) or to a
  persistent flush buffer (streamed).
- **Frozen params still get accumulators.** Both `Stage` and `OffloadModel`
  allocate `grad_acc` eagerly for ALL `named_parameters()`, ignoring
  `requires_grad` — and under `grad_accum="stream"` the CPU ones are PINNED.
  On the frozen 4B encoder that is ~8 GB of host RAM; on a frozen 12.8B
  resident DiT stage ~7 GB of VRAM. For forward-only models reclaim it with
  `utils/ramtorch_helpers.drop_grad_accumulators` (post-construction) or
  `no_grad_accumulators()` (during construction, for resident stages where the
  allocation happens in `Stage.__init__`). You cannot reclaim it for a model
  you actually train.
- Bare `torch.utils.checkpoint` INSIDE a chunk is only valid for resident
  stages — under the offload engine `functional_call` reverts the swapped-in
  GPU weights on exit, so it recomputes against the CPU masters. Use
  `keep_activations="checkpoint"` instead (the trainer raises if you set
  `grad_ckpt` with a streamed parallelism).
- Deepcopy a model BEFORE constructing an offload engine on it: the ctor
  relocates chunk params in place (CPU pinned / NVMe-mapped).
- Offload full FT of a 12.8B model is bound by **CPU optimizer work**, not
  PCIe: `flush_grads` + `clip_grad_norm_` + fused CPU AdamW + `zero_grad_acc`
  touch ~400 GB of host memory per step (~35 s), while `acquire_wait_s` stays
  ~0.04 s. LoRA mode (bf16 masters, ~119M optimizer params) is ~4.4 s/step.
- CPU-side stall counters mislead in offload mode: `acquire_wait_s` and
  `wait L{k}` spans stay near zero even when the GPU is idle, because a chunk
  counts as resident once its copy is *enqueued* and the compute stream then
  waits on a CUDA event (device-side, invisible to CPU annotations). Judge
  overlap by GPU busy time (sum of `kernel` durations) vs the wall span.

# Fused Lion optimizer — standalone CPU kernel

A single-file, dependency-free (just libtorch) fused Lion optimizer step for CPU,
for when you want DeepSpeedCPULion-style performance without the DeepSpeed stack.

## Algorithm

Lion — *Symbolic Discovery of Optimization Algorithms* (Chen et al., 2023,
[arXiv:2302.06675](https://arxiv.org/abs/2302.06675)):

```
interp = m * beta1 + g * (1 - beta1)     # uses OLD momentum
p      = p * (1 - lr*wd) - lr * sign(interp)
m      = m * beta2 + g * (1 - beta2)
```

`sign(0) = 0` (paper / lion-pytorch semantics; note DeepSpeed's kernel uses
`copysign` so exact zeros round to +1 — differs only on exact zeros).

## Files

- `fused_lion.cpp` — the kernel. Structure adapted from PyTorch's
  `aten/src/ATen/native/cpu/FusedAdamKernel.cpp` (BSD — © Meta/PyTorch):
  `at::vec::Vectorized` SIMD with fp32 opmath for bf16/fp16 inputs, fp32
  computation everywhere, cache-aware chunking, one `at::parallel_for` across
  the whole parameter list (multi-tensor: one thread pool dispatch per step).
  Algorithmic reference: DeepSpeed `csrc/lion/` (Apache-2.0 — © Microsoft).
- `bench_fused_lion.py` — builds the extension, checks parity, benchmarks
  against python-op / foreach / torch.compile lion.

This directory is Apache-2.0 (compatible with both upstream sources).

## Usage

```python
from torch.utils.cpp_extension import load
ext = load(name="fused_lion_cpu", sources=["fused_lion.cpp"],
           extra_cflags=["-O3", "-march=native", "-ffp-contract=off",
                         "-DCPU_CAPABILITY_AVX512"],  # if CPU has AVX512
           build_directory="build")

ext.fused_lion_(params, grads, moments, lr, beta1, beta2, weight_decay)
```

- params/grads: fp32, bf16 or fp16; moments: param dtype or fp32.
- all tensors CPU + contiguous, uniform dtypes across the list.
- in place; releases the GIL during the step.

**Gotcha (found while testing):** with plain extension flags ATen's vec
namespace compiles down to AVX2 (vec width 8) even with `-march=native` —
ATen selects its SIMD level from the `CPU_CAPABILITY_*` macros, not
`__AVX512F__`. Pass `-DCPU_CAPABILITY_AVX512` on AVX512 hardware (the bench
script probes `ext.vec_size()` and rebuilds automatically).

## Results (Ryzen 5 7600, 6C/12T Zen4 + AVX-512, torch 2.14, 6 threads)

300M params (120 × 2.5M), fp32, best of 20 in-place steps:

| impl | ms/step | vs fused | effective GB/s |
|---|---|---|---|
| **fused (this kernel)** | **181** | 1.00x | ~33 |
| torch.compile lion | 164 | 0.90x | ~37 |
| python ops (lucidrains-style) | 351 | 1.93x | ~17 |
| foreach | 604 | 3.33x | ~10 |
| fused, bf16 params + fp32 states | 146 | 0.81x | — (moves 30% fewer bytes) |

- fp32 parity vs a non-fused torch-op reference: **bitwise identical**
  (`-ffp-contract=off`, same op order). bf16/fp16 params match reference to
  within quantization.
- Both fused and torch.compile are DRAM-bandwidth-bound (12 threads gives no
  speedup), so on a server board (8-channel EPYC/Xeon) absolute numbers scale
  up but the ranking stays.
- Honest note: `torch.compile` matches the hand-written kernel for the plain
  fp32 case (both fuse to a single C++ pass). The kernel still buys you: no
  compile warmup or dynamo guards, one call per step for the whole param
  list, and bf16/fp16 params with fp32 *or* low-precision states — which is
  the config you want for optimizer-offload (Lion state = 1 buffer, half of
  Adam's, halved again if you keep it bf16).

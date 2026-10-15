// Fused Lion optimizer kernel for CPU — standalone torch extension.
//
// Algorithm: Lion — "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023,
// https://arxiv.org/abs/2302.06675):
//     interp = m * beta1 + g * (1 - beta1)      # uses OLD momentum
//     p      = p * (1 - lr*wd) - lr * sign(interp)
//     m      = m * beta2 + g * (1 - beta2)
//   sign(0) = 0 (matches the paper and lion-pytorch; note DeepSpeed's copysign-based
//   kernel gives sign(0) = +1 — only differs on exact zeros).
//
// Kernel structure adapted from PyTorch's aten/src/ATen/native/cpu/FusedAdamKernel.cpp
// (BSD-style, Copyright (c) Meta Platforms / PyTorch contributors): at::vec SIMD with
// fp32 opmath for reduced-precision inputs, cache-aware work splitting, at::parallel_for.
// Algorithmic reference for the CPU optimizer: DeepSpeedCPULion
// (deepspeed/csrc/lion/, Apache-2.0, Copyright (c) Microsoft Corporation).
//
// This file: Apache-2.0 (compatible with both upstream licenses).

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int64_t kGrain = 16384; // elements per work chunk (~64KB fp32)

inline float sign0(float v) { return v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); }

template <typename fVec>
inline fVec sign_vec(const fVec& v) {
  auto zero = fVec(0.0f);
  auto pos = v > zero;
  auto neg = v < zero;
  return fVec::blendv(fVec::blendv(zero, fVec(1.0f), pos), fVec(-1.0f), neg);
}

// fp32 math core; updates param and momentum in place.
template <typename fVec>
inline void lion_core(fVec& p, const fVec& g, fVec& m, float lr, float b1, float b2,
                      float decay, float neg_lr) {
  auto interp = m * fVec(b1) + g * fVec(1.0f - b1);
  auto u = sign_vec(interp);
  if (decay != 1.0f) p = p * fVec(decay);
  p = p + u * fVec(neg_lr);
  m = m * fVec(b2) + g * fVec(1.0f - b2);
}

struct LionScalars {
  float lr, b1, b2, decay, neg_lr;
  LionScalars(double lr_, double b1_, double b2_, double wd)
      : lr((float)lr_),
        b1((float)b1_),
        b2((float)b2_),
        decay((float)(1.0 - lr_ * wd)),
        neg_lr(-(float)lr_) {}
};

// ---------------------------------------------------------------- fp32 x fp32
template <typename scalar_t, typename state_t,
          std::enable_if_t<std::is_same_v<scalar_t, float> &&
                               std::is_same_v<state_t, float>,
                           int> = 0>
inline void lion_kernel(const scalar_t* g, scalar_t* p, state_t* m, int64_t size,
                        const LionScalars& s) {
  using Vec = at::vec::Vectorized<float>;
  int64_t d = 0;
  for (; d + Vec::size() <= size; d += Vec::size()) {
    auto pv = Vec::loadu(p + d);
    auto gv = Vec::loadu(g + d);
    auto mv = Vec::loadu(m + d);
    lion_core(pv, gv, mv, s.lr, s.b1, s.b2, s.decay, s.neg_lr);
    pv.store(p + d);
    mv.store(m + d);
  }
  for (; d < size; d++) {
    float pv = p[d], gv = g[d], mv = m[d];
    float interp = mv * s.b1 + gv * (1.0f - s.b1);
    if (s.decay != 1.0f) pv *= s.decay;
    pv += s.neg_lr * sign0(interp);
    m[d] = mv * s.b2 + gv * (1.0f - s.b2);
    p[d] = pv;
  }
}

// --------------------------------------------- lp params (bf16/fp16) x (fp32 or lp) state
template <typename scalar_t, typename state_t,
          std::enable_if_t<!std::is_same_v<scalar_t, float> &&
                               (std::is_same_v<state_t, float> ||
                                std::is_same_v<state_t, scalar_t>),
                           int> = 0>
inline void lion_kernel(const scalar_t* g, scalar_t* p, state_t* m, int64_t size,
                        const LionScalars& s) {
  using lpVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr bool fp32_state = std::is_same_v<state_t, float>;
  constexpr int64_t half = lpVec::size() / 2; // == fVec::size()

  int64_t d = 0;
  for (; d + lpVec::size() <= size; d += lpVec::size()) {
    auto [g1, g2] = at::vec::convert_to_float<scalar_t>(lpVec::loadu(g + d));
    auto [p1, p2] = at::vec::convert_to_float<scalar_t>(lpVec::loadu(p + d));
    fVec m1, m2;
    if constexpr (fp32_state) {
      m1 = fVec::loadu(m + d);
      m2 = fVec::loadu(m + d + half);
    } else {
      auto [mm1, mm2] = at::vec::convert_to_float<scalar_t>(lpVec::loadu(m + d));
      m1 = mm1;
      m2 = mm2;
    }
    lion_core(p1, g1, m1, s.lr, s.b1, s.b2, s.decay, s.neg_lr);
    lion_core(p2, g2, m2, s.lr, s.b1, s.b2, s.decay, s.neg_lr);
    if constexpr (fp32_state) {
      m1.store(m + d);
      m2.store(m + d + half);
    } else {
      at::vec::convert_from_float<scalar_t>(m1, m2).store(m + d);
    }
    at::vec::convert_from_float<scalar_t>(p1, p2).store(p + d);
  }
  for (; d < size; d++) {
    float pv = (float)p[d], gv = (float)g[d], mv = (float)m[d];
    float interp = mv * s.b1 + gv * (1.0f - s.b1);
    if (s.decay != 1.0f) pv *= s.decay;
    pv += s.neg_lr * sign0(interp);
    m[d] = (state_t)(mv * s.b2 + gv * (1.0f - s.b2));
    p[d] = (scalar_t)pv;
  }
}

template <typename param_t, typename state_t>
void run_all(const std::vector<const void*>& gs, const std::vector<void*>& ps,
             const std::vector<void*>& ms, const std::vector<int64_t>& numels,
             const std::vector<int64_t>& chunk_off, int64_t total_chunks,
             const LionScalars& s) {
  at::parallel_for(0, total_chunks, 1, [&](int64_t begin, int64_t end) {
    int64_t ti = 0;
    while (chunk_off[ti + 1] <= begin) ti++;
    for (int64_t c = begin; c < end; c++) {
      while (chunk_off[ti + 1] <= c) ti++;
      int64_t start = (c - chunk_off[ti]) * kGrain;
      int64_t len = std::min(start + kGrain, numels[ti]) - start;
      lion_kernel<param_t, state_t>((const param_t*)gs[ti] + start,
                                    (param_t*)ps[ti] + start,
                                    (state_t*)ms[ti] + start, len, s);
    }
  });
}

template <typename param_t>
void run_all_dispatch(const std::vector<const void*>& gs, const std::vector<void*>& ps,
                      const std::vector<void*>& ms, const std::vector<int64_t>& numels,
                      bool fp32_state, const std::vector<int64_t>& chunk_off,
                      int64_t total_chunks, const LionScalars& s) {
  if (fp32_state)
    run_all<param_t, float>(gs, ps, ms, numels, chunk_off, total_chunks, s);
  else
    run_all<param_t, param_t>(gs, ps, ms, numels, chunk_off, total_chunks, s);
}

void fused_lion_(std::vector<torch::Tensor> params, std::vector<torch::Tensor> grads,
                 std::vector<torch::Tensor> exp_avgs, double lr, double beta1,
                 double beta2, double weight_decay) {
  TORCH_CHECK(params.size() == grads.size() && params.size() == exp_avgs.size(),
              "params/grads/exp_avgs length mismatch");
  const int64_t nt = (int64_t)params.size();
  if (nt == 0) return;

  std::vector<const void*> gs(nt);
  std::vector<void*> ps(nt), ms(nt);
  std::vector<int64_t> numels(nt), chunk_off(nt + 1, 0);
  int64_t total_chunks = 0;
  bool fp32_state = false;

  for (int64_t i = 0; i < nt; i++) {
    auto& p = params[i];
    auto& g = grads[i];
    auto& m = exp_avgs[i];
    TORCH_CHECK(p.is_cpu() && g.is_cpu() && m.is_cpu(), "all tensors must be CPU");
    TORCH_CHECK(p.numel() == g.numel() && p.numel() == m.numel(), "numel mismatch at ", i);
    TORCH_CHECK(p.scalar_type() == g.scalar_type(), "grad dtype must match param at ", i);
    TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous(),
                "tensors must be contiguous at ", i);
    TORCH_CHECK(m.scalar_type() == p.scalar_type() || m.scalar_type() == torch::kFloat,
                "state must be param dtype or fp32 at ", i);
    TORCH_CHECK(p.numel() > 0, "empty tensor at ", i);
    if (i == 0) fp32_state = m.scalar_type() == torch::kFloat;
    else TORCH_CHECK((m.scalar_type() == torch::kFloat) == fp32_state,
                     "state dtype must be uniform across tensors");
    gs[i] = g.data_ptr();
    ps[i] = p.data_ptr();
    ms[i] = m.data_ptr();
    numels[i] = p.numel();
    total_chunks += (numels[i] + kGrain - 1) / kGrain;
    chunk_off[i + 1] = total_chunks;
  }

  const LionScalars s(lr, beta1, beta2, weight_decay);
  auto st = params[0].scalar_type();
  if (st == torch::kFloat) {
    run_all<float, float>(gs, ps, ms, numels, chunk_off, total_chunks, s);
  } else if (st == torch::kBFloat16) {
    run_all_dispatch<c10::BFloat16>(gs, ps, ms, numels, fp32_state, chunk_off,
                                    total_chunks, s);
  } else if (st == torch::kHalf) {
    run_all_dispatch<c10::Half>(gs, ps, ms, numels, fp32_state, chunk_off,
                                total_chunks, s);
  } else {
    TORCH_CHECK(false, "unsupported param dtype ", st);
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_lion_", &fused_lion_,
        "Fused Lion optimizer step on CPU (in place)",
        py::arg("params"), py::arg("grads"), py::arg("exp_avgs"), py::arg("lr"),
        py::arg("beta1"), py::arg("beta2"), py::arg("weight_decay"),
        py::call_guard<py::gil_scoped_release>());
  m.def("vec_size", []() { return (int64_t)at::vec::Vectorized<float>::size(); },
        "SIMD width of ATen vec in this build (16 = AVX512, 8 = AVX2, 4 = SSE)");
}

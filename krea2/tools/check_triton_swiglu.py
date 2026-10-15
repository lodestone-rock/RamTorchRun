"""check_triton_swiglu.py — Is the vendored Axolotl Triton SwiGLU safe to train with?

Verifies krea2/model/triton_swiglu.py (opt-in via KREA2_TRITON_SWIGLU=1 or
mmdit.set_triton_swiglu(True)) against the eager SwiGLU it replaces:

  1. plain autograd: bit-exact forward, gradients within bf16 noise;
  2. the live SAC path (checkpoint_block + the run's recompute list): same
     checks, plus retained-activation memory after forward vs eager;
  3. quick timing (fwd+bwd), informative only.

Run on a free GPU: uv run python krea2/tools/check_triton_swiglu.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time

import torch

from krea2.checkpointing import SelectiveCheckpointOptions, checkpoint_block
from krea2.model.mmdit import SwiGLU, set_triton_swiglu
from krea2.model.triton_swiglu import _REGISTERED

TOKENS = (4096, 1237)  # 32x128 batch shape + an odd one
FEATURES, MULTIPLIER = 6144, 4  # krea2 large_wide -> mlpdim 16384
SAC_RECOMPUTE = frozenset({"attn.qkv", "attn.gate", "attn.out"})  # live k2-lion config


def mb(nbytes: float) -> str:
    return f"{nbytes / 1024 / 1024:.0f}MiB"


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    stable = b.abs() > 0.1 * (b.abs().max().item() + 1e-9)
    if not stable.any():
        return (a - b).abs().max().item()
    return ((a - b).abs() / b.abs().clamp_min(1e-9))[stable].max().item()


def compare(device: str, sac: bool) -> bool:
    ok = True
    for tokens in TOKENS:
        torch.manual_seed(0)
        m = SwiGLU(FEATURES, MULTIPLIER).to(device).to(torch.bfloat16)
        if sac:
            m._selective_checkpoint_options = SelectiveCheckpointOptions(
                recompute=SAC_RECOMPUTE
            )
            m._selective_checkpoint_enabled = True
        for p in m.parameters():
            p.requires_grad_(True)
        x = (torch.randn(tokens, FEATURES, device=device) * 0.5).to(torch.bfloat16)
        g = torch.randn(tokens, FEATURES, device=device).to(torch.bfloat16)

        results = {}
        for label, enabled in (("eager", False), ("triton", True)):
            set_triton_swiglu(enabled)
            m.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            xin = x.clone().requires_grad_(True)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            out = checkpoint_block(m, xin) if sac else m(xin)
            torch.cuda.synchronize()
            retained = torch.cuda.memory_allocated() - base - xin.numel() * 2 \
                - out.numel() * 2
            fwd_bitwise = None
            if label == "eager":
                ref_out, ref_grads = out.detach(), None
            else:
                fwd_bitwise = bool(torch.equal(out, ref_out))
            out.backward(g)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
            results[label] = {
                "out": out.detach(),
                "grads": {"x": xin.grad.clone(),
                          **{n: p.grad.clone() for n, p in m.named_parameters()}},
                "retained": retained, "peak": peak, "fwd_bitwise": fwd_bitwise,
            }
            torch.cuda.empty_cache()

        r, e = results["triton"], results["eager"]
        grad_errs = {k: rel_err(v, e["grads"][k]) for k, v in r["grads"].items()}
        worst = max(grad_errs.values())
        ok_case = r["fwd_bitwise"] and worst < 5e-2
        ok &= ok_case
        tag = f"{'SAC' if sac else '   '} tok={tokens}"
        print(f"  [{tag}] fwd bitwise={'YES' if r['fwd_bitwise'] else 'NO '}  "
              f"worst grad rel={worst:.2e}")
        print(f"          retained eager={mb(e['retained'])} triton={mb(r['retained'])}  "
              f"peak eager={mb(e['peak'])} triton={mb(r['peak'])}")
        if not ok_case:
            for k, v in grad_errs.items():
                print(f"          {k}: rel={v:.2e}")
        set_triton_swiglu(False)
    return ok


def bench(device: str) -> None:
    tokens = TOKENS[0]
    for label, enabled in (("eager", False), ("triton", True)):
        set_triton_swiglu(enabled)
        m = SwiGLU(FEATURES, MULTIPLIER).to(device).to(torch.bfloat16)
        x = (torch.randn(tokens, FEATURES, device=device) * 0.5).to(torch.bfloat16)
        g = torch.randn(tokens, FEATURES, device=device).to(torch.bfloat16)
        for _ in range(5):
            m(x).backward(g)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            m(x).backward(g)
        torch.cuda.synchronize()
        print(f"  bench[{label:6s}] {(time.perf_counter() - t0) / 20 * 1e3:6.2f} ms/iter")
        torch.cuda.empty_cache()
    set_triton_swiglu(False)


def main() -> None:
    assert torch.cuda.is_available(), "needs a GPU (triton kernels)"
    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}   dispatcher ops registered: {_REGISTERED}")
    print(f"dims: features={FEATURES} multiplier={MULTIPLIER} "
          f"-> mlpdim={int(2 * FEATURES / 3) * MULTIPLIER}")
    print(f"sac recompute list: {sorted(SAC_RECOMPUTE)} (live k2-lion config)\n")

    print("1. plain autograd, triton toggle vs eager")
    ok = compare(device, sac=False)
    print("\n2. SAC path (checkpoint_block + selective checkpoint), triton vs eager")
    ok &= compare(device, sac=True)

    print("\n3. timing (informative)")
    bench(device)

    print(f"\n{'PASS' if ok else 'FAIL'}: triton SwiGLU "
          f"{'matches' if ok else 'MISMATCHES'} eager")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

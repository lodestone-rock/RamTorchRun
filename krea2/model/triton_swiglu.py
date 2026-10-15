"""Triton SwiGLU kernels for Krea-2, vendored from Axolotl.

Upstream: https://github.com/axolotl-ai-cloud/axolotl
          src/axolotl/kernels/{swiglu,op_registry}.py @ main, 2026-09
License:  Apache-2.0 (c) Axolotl AI contributors.
Credit:   Axolotl team; kernel design inspired by Unsloth (https://unsloth.ai),
          per the upstream module docstring. Math per "GLU Variants Improve
          Transformer" (https://arxiv.org/abs/2002.05202).

Local changes vs upstream (kernels themselves are verbatim):
- `swiglu_forward` allocates via `torch.empty_like` instead of unpacking a 3D
  shape, so any leading dims work (Krea feeds 2D packed tokens).
- Registered as dispatcher ops under the `krea2::` namespace so torch's
  selective-checkpoint policy can see them (PREFER_RECOMPUTE, same as the
  eager silu/mul they replace). Falls back to raw callables if registration
  fails or the ops already exist (module re-import).
- `triton_silu_mul` guards: CUDA + floating + same shape/dtype + contiguous;
  otherwise silently computes the eager `F.silu(gate) * up` (keeps CPU
  chunk-parity tests and non-CUDA paths working).

Backward is in-place by contract (upstream design): the bwd kernel overwrites
`grad_output` with the recomputed forward output, `gate` with d(gate), and
`up` with d(up) — free recompute, zero extra allocations. This means the op
supports exactly one backward per forward: no retain_graph, no double-backward.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NAMESPACE = "krea2"
_warned_registration_failure = False


# ---------------------------------------------------------------------------
# begin upstream swiglu.py kernels (verbatim)
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load gate in fp32, keep up in original dtype
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)

    # Compute activation in fp32 then convert back
    f = gate * tl.sigmoid(gate)
    f = f.to(up.dtype)
    result = f * up

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    grad_out_ptr,
    gate_ptr,
    up_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    """After kernel execution, tensors are modified in-place:
    - `grad_out_ptr` contains forward output (`h`)
    - `gate_ptr` contains gradient w.r.t gate (`grad_gate`)
    - `up_ptr` contains gradient w.r.t up (`grad_up`)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load values - only convert gate to fp32
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0)

    # Compute SiLU and forward output
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = sigmoid_gate * gate
    silu_gate = silu_gate.to(grad_out.dtype)
    h = silu_gate * up

    # Compute gradients
    grad_up = grad_out * silu_gate  # gradient for up is grad_out * SiLU(gate)

    # Compute gate gradient
    temp = grad_out * up
    grad_gate = temp.to(tl.float32) * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    grad_gate = grad_gate.to(grad_out.dtype)

    # Store results with correct gradient ordering
    tl.store(grad_out_ptr + offsets, h, mask=mask)
    tl.store(gate_ptr + offsets, grad_gate, mask=mask)  # grad wrt gate
    tl.store(up_ptr + offsets, grad_up, mask=mask)  # grad wrt up
# ---------------------------------------------------------------------------
# end upstream kernels
# ---------------------------------------------------------------------------


def _swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    n_elements = gate.numel()
    out = torch.empty_like(gate)

    grid = lambda meta: (triton.cdiv(n_elements, meta["block_size"]),)  # noqa: E731
    _swiglu_fwd_kernel[grid](
        gate_ptr=gate,
        up_ptr=up,
        out_ptr=out,
        n_elements=n_elements,
        block_size=1024,
    )
    return out


def _swiglu_backward_inplace(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """In-place backward: returns (h, dgate, dup); mutates its arguments."""
    n_elements = grad_output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["block_size"]),)  # noqa: E731
    _swiglu_bwd_kernel[grid](
        grad_out_ptr=grad_output,
        gate_ptr=gate,
        up_ptr=up,
        n_elements=n_elements,
        block_size=1024,
    )
    return grad_output, gate, up


# --- dispatcher registration (upstream op_registry pattern) -----------------
# Registers the kernels as `krea2::swiglu_fwd` / `krea2::swiglu_bwd` so
# dispatch-mode machinery (selective checkpointing, torch.compile) sees them
# as ordinary recomputable ops instead of opaque Python. On any failure the
# raw callables keep working — the op just stays invisible to the dispatcher.
_REGISTERED = False
try:
    if not hasattr(torch.ops.krea2, "swiglu_fwd"):
        @torch.library.custom_op(f"{NAMESPACE}::swiglu_fwd", mutates_args=())
        def _op_swiglu_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return _swiglu_forward(gate, up)

        @_op_swiglu_fwd.register_fake
        def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(gate)

    if not hasattr(torch.ops.krea2, "swiglu_bwd"):
        @torch.library.custom_op(f"{NAMESPACE}::swiglu_bwd",
                                 mutates_args=("grad_output", "gate", "up"))
        def _op_swiglu_bwd(
            grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
        ) -> None:
            _swiglu_backward_inplace(grad_output, gate, up)

        @_op_swiglu_bwd.register_fake
        def _(
            grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
        ) -> None:
            return None

    _REGISTERED = hasattr(torch.ops.krea2, "swiglu_fwd") and hasattr(
        torch.ops.krea2, "swiglu_bwd"
    )
except Exception as e:  # pragma: no cover - exotic torch builds
    _REGISTERED = False
    _warn_msg = f"krea2.triton_swiglu: dispatcher registration failed ({e}); using raw callables"


def _fwd_call(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if _REGISTERED:
        return torch.ops.krea2.swiglu_fwd(gate, up)
    return _swiglu_forward(gate, up)


def _bwd_call(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> None:
    if _REGISTERED:
        torch.ops.krea2.swiglu_bwd(grad_output, gate, up)
    else:
        _swiglu_backward_inplace(grad_output, gate, up)


class TritonSwiGLUFunction(torch.autograd.Function):
    """silu(gate) * up with the Axolotl Triton fwd/bwd kernels.

    Saves gate/up; backward overwrites them in place (upstream contract), so
    the op is single-backward: no retain_graph / double-backward.
    """

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        h = _fwd_call(gate, up)
        ctx.save_for_backward(gate, up)
        return h

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        dc = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
        _bwd_call(dc, gate, up)
        # dc now holds h (discarded); gate/up now hold the gradients.
        return gate, up


_warned_fallback = False


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up via the vendored Axolotl Triton kernels when eligible.

    Falls back to eager `F.silu(gate) * up` on CPU / non-floating / shape or
    dtype mismatch / non-contiguous inputs, so callers never have to guard.
    """
    global _warned_fallback
    if (
        gate.is_cuda
        and gate.is_floating_point()
        and up.is_cuda
        and gate.dtype == up.dtype
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
    ):
        return TritonSwiGLUFunction.apply(gate, up)
    if not gate.is_cuda:
        return F.silu(gate) * up  # intentional: CPU parity/eager paths stay silent
    if not _warned_fallback:
        warnings.warn(
            "krea2.triton_swiglu: inputs not eligible (dtype/shape/contiguity); "
            "using eager silu*mul", stacklevel=2,
        )
        _warned_fallback = True
    return F.silu(gate) * up

"""Batch-1 LoRA speed paths for the MOSS DiT: fused q/k/v and FP8 frozen-base matmuls.

Both keep the adapter parameters where ``inject_lora`` put them (``layers.N.proj.q.lora_A`` ...), so state
dicts, exports and optimizers are unchanged; only the frozen base matmul and the call structure differ.

``fuse_qkv``: one [3d, d] base GEMM per block instead of three, and one [3r, d] adapter down-projection (x read
once); the three up-projections stay separate (block-diagonal).

``enable_fp8``: the frozen base weights are quantized once to float8_e4m3 (forward: per output row; input
gradient: per input column), activations and incoming gradients per token, through ``torch._scaled_mm``. The
bf16 base weights are dropped. The adapters stay bf16/fp32. ``scaling`` "rowwise" | "tensorwise".
"""
from __future__ import annotations

import types

import torch
from torch import nn
from torch.nn import functional as F

from moss.model.lora import LoRALinear
from moss.model.transformer import attention, situ

E4M3 = torch.float8_e4m3fn
E4M3_MAX = torch.finfo(E4M3).max


def quantize(x: torch.Tensor, rowwise: bool):
    """[M, K] -> (float8 [M, K], fp32 scale [M, 1] rowwise or [] tensorwise), x ~= q * scale."""
    x = x.float()
    amax = x.abs().amax(-1, keepdim=True) if rowwise else x.abs().amax()
    scale = amax.clamp(min=1e-12) / E4M3_MAX
    return (x / scale).to(E4M3), scale


class FP8Weight(nn.Module):
    """A frozen [N, K] weight held as two float8 copies: W (rows = outputs) for y = x W^T and W^T (rows =
    inputs) for dx = dy W, each scaled along the matmul's reduction-free axis."""

    def __init__(self, weight: torch.Tensor, rowwise: bool = True):
        super().__init__()
        self.rowwise = rowwise
        w = weight.detach()
        q, s = quantize(w, rowwise)
        qt, st = quantize(w.t().contiguous(), rowwise)
        self.register_buffer("w", q, persistent=False)
        self.register_buffer("w_scale", s.reshape(1, -1) if rowwise else s, persistent=False)
        self.register_buffer("wt", qt, persistent=False)
        self.register_buffer("wt_scale", st.reshape(1, -1) if rowwise else st, persistent=False)
        self.out_features, self.in_features = w.shape

    def forward(self, x):
        out = fp8_linear(x.flatten(0, -2), self.w, self.w_scale, self.wt, self.wt_scale, self.rowwise)
        return out.unflatten(0, x.shape[:-1])


# Custom ops, opaque to the outer torch.compile: an autograd.Function here trips dynamo when two projections
# share an input (torch 2.10, "lift_tracked_freevar_to_input"). The quantize inside gets its own compiled kernel.
_quantize = torch.compile(quantize, dynamic=True)


def _mm(a, a_scale, b, b_scale):
    """a [M, K] float8 row-major, b [N, K] float8 row-major -> a b^T in bf16."""
    return torch._scaled_mm(a, b.t(), scale_a=a_scale, scale_b=b_scale, out_dtype=torch.bfloat16)


@torch.library.custom_op("moss::fp8_linear", mutates_args=())
def fp8_linear(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, wt: torch.Tensor, wt_scale: torch.Tensor,
               rowwise: bool) -> torch.Tensor:
    xq, xs = _quantize(x, rowwise)
    return _mm(xq, xs, w, w_scale)


@fp8_linear.register_fake
def _(x, w, w_scale, wt, wt_scale, rowwise):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("moss::fp8_linear_grad", mutates_args=())
def fp8_linear_grad(grad: torch.Tensor, wt: torch.Tensor, wt_scale: torch.Tensor, rowwise: bool) -> torch.Tensor:
    gq, gs = _quantize(grad, rowwise)
    return _mm(gq, gs, wt, wt_scale)


@fp8_linear_grad.register_fake
def _(grad, wt, wt_scale, rowwise):
    return grad.new_empty((grad.shape[0], wt.shape[0]), dtype=torch.bfloat16)


def _setup(ctx, inputs, output):
    _, _, _, wt, wt_scale, rowwise = inputs
    ctx.save_for_backward(wt, wt_scale)
    ctx.rowwise = rowwise


def _backward(ctx, grad):
    wt, wt_scale = ctx.saved_tensors
    return fp8_linear_grad(grad.contiguous(), wt, wt_scale, ctx.rowwise), None, None, None, None, None


fp8_linear.register_autograd(_backward, setup_context=_setup)


def _adapter(m: LoRALinear, x):
    ab = m.adapter(m.active_role)
    if ab is None:
        return None
    A, B = ab
    return F.linear(F.linear(x, A), B) * m.scale


def _fp8_lora_forward(self, x):
    y = self.fp8(x)
    if self.base_bias is not None:
        y = y + self.base_bias
    delta = _adapter(self, x)
    return y if delta is None else y + delta


class FP8LoRALinear(LoRALinear):
    forward = _fp8_lora_forward


def enable_fp8(model: nn.Module, rowwise: bool = True) -> int:
    """Quantize the base of every LoRALinear (and every fused q/k/v base) to float8; drop the bf16 copies."""
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.base_weight.numel():
            # A 0-size bf16 stub keeps ``weight.dtype`` (read by Block.forward) answering.
            stub = module.base_weight.new_empty(0)
            if not getattr(module, "fused", False):
                module.fp8 = FP8Weight(module.base_weight, rowwise)
                module.__class__ = FP8LoRALinear
                n += 1
            module.base_weight = stub
        elif hasattr(module, "qkv_base") and isinstance(module.qkv_base, torch.Tensor):
            module.qkv_fp8 = FP8Weight(module.qkv_base, rowwise)
            module.qkv_base = None
            n += 1
    if not n:
        raise ValueError("no LoRALinear modules found")
    return n


def _fused_block_forward(self, x, rope, modulation, mask=None):
    """Block.forward with one q/k/v base GEMM and one adapter down-projection."""
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
    q_m, k_m, v_m = self.proj["q"], self.proj["k"], self.proj["v"]
    y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
    dtype = self.proj["o"].weight.dtype
    y = y.to(dtype)
    qkv = self.qkv_fp8(y) if self.qkv_base is None else F.linear(y, self.qkv_base)
    q, k, v = qkv.chunk(3, -1)
    roles = [m.adapter(m.active_role) for m in (q_m, k_m, v_m)]
    if roles[0] is not None:
        down = F.linear(y, torch.cat([ab[0] for ab in roles]))
        hq, hk, hv = down.chunk(3, -1)
        q = q + F.linear(hq, roles[0][1]) * q_m.scale
        k = k + F.linear(hk, roles[1][1]) * k_m.scale
        v = v + F.linear(hv, roles[2][1]) * v_m.scale
    q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
    k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
    x = x + self.proj["o"](attention(q, k, v, rope, self.heads, mask)) * gate_a[:, None]
    y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
    hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


def fuse_qkv(model: nn.Module) -> int:
    """Every block whose q/k/v are LoRALinear (same rank, no frozen adapter, no bias) runs the fused forward.
    The separate q/k/v bases become views into the fused one (no extra memory), so call it after the model is
    on its device."""
    n = 0
    for block in model.layers:
        q_m, k_m, v_m = (block.proj[name] for name in ("q", "k", "v"))
        if not all(isinstance(m, LoRALinear) for m in (q_m, k_m, v_m)):
            continue
        if any(m.base_bias is not None or m.frozen_A is not None for m in (q_m, k_m, v_m)):
            raise ValueError("fuse_qkv supports bias-free q/k/v without a frozen adapter")
        fused = torch.cat([q_m.base_weight, k_m.base_weight, v_m.base_weight])
        block.register_buffer("qkv_base", fused, persistent=False)
        d = q_m.out_features
        for i, m in enumerate((q_m, k_m, v_m)):
            m.base_weight, m.fused = fused[i * d:(i + 1) * d], True
        block.forward = types.MethodType(_fused_block_forward, block)
        n += 1
    return n

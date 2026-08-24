"""RoPE + attention math for Chroma (ported from lodestone-rock/flow).

`attention` optionally wraps SDPA in ``sdpa_kernel(CUDNN)``. That context
saves/restores PROCESS-GLOBAL backend flags and is not thread-safe —
multithreaded trainers (RamTorch pipeline stage workers) must call
``set_sdpa_ctx(False)`` and pin the global backend priority once at startup
instead (see ``chroma/train_utils.py::_pin_sdpa_backends``). It also has no
CPU backend, so the CPU parity tools disable it too.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

_sdpa_ctx_enabled = True


def set_sdpa_ctx(enabled: bool):
    global _sdpa_ctx_enabled
    _sdpa_ctx_enabled = enabled


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask: Tensor | None = None) -> Tensor:
    q, k = apply_rope(q, k, pe)
    if _sdpa_ctx_enabled:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return rearrange(x, "B H L D -> B L (H D)")


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

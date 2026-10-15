"""ConvRot W8A8: group-wise regular-Hadamard rotation + INT8 weights/activations.

ConvRot (arXiv 2512.03673) splits a linear's K input channels into blocks of
`gs` (a power of 4) and rotates each block by a *regular* Hadamard matrix
(every row/column sums to +-sqrt(gs)), built as Kronecker powers of H4. The
rotation is orthogonal, so `rot(x) @ rot(W).T == x @ W.T`; weights are rotated
once offline, activations online at every call. It flattens row-wise outliers
before per-token INT8 quantization instead of amplifying them the way the
Sylvester (FWHT) Hadamard does.

Storage is stock ComfyUI's native `int8_tensorwise` format (ComfyUI >= 0.27,
comfy-kitchen `TensorWiseINT8Layout`), so one file serves both ComfyUI and
`krea2/inference.py`:

    <layer>.weight        int8 [out, in]   rotated weight
    <layer>.weight_scale  f32  [out, 1]    per-output-row scale
    <layer>.comfy_quant   uint8 JSON       {"format": "int8_tensorwise",
                                            "convrot": true, "convrot_groupsize": 256}

The math below is a port of comfy-kitchen 0.2.37 (`int8_utils.py`,
`backends/eager/quantization.py::int8_linear`), not an import: the H4 sign
pattern, Kronecker order and quantizer rounding must match it exactly or a
checkpoint written here decodes to garbage in ComfyUI.
"""
from __future__ import annotations

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

COMFY_FORMAT = "int8_tensorwise"
MARKER_SUFFIX = ".comfy_quant"

_H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
_HADAMARD_CACHE: dict[tuple, Tensor] = {}
CLIP_GRID = torch.linspace(0.55, 1.0, 80)


def regular_hadamard(n: int, device="cpu", dtype=torch.float32) -> Tensor:
    """Normalized regular Hadamard of order n = 4^k: H4 kron H4 kron ... / sqrt(n).
    Symmetric and orthogonal, so it is its own inverse."""
    key = (n, str(device), dtype)
    if key not in _HADAMARD_CACHE:
        if n < 4 or (n & (n - 1)) != 0 or math.log(n, 4) % 1 != 0:
            raise ValueError(f"Regular Hadamard size must be a power of 4, got {n}")
        h4 = torch.tensor(_H4, dtype=dtype, device=device)
        h = h4
        while h.shape[0] < n:
            h = torch.kron(h, h4)
        _HADAMARD_CACHE[key] = h / (n ** 0.5)
    return _HADAMARD_CACHE[key]


def rotate_weight(w: Tensor, h: Tensor, gs: int) -> Tensor:
    """Offline: W_rot = W_blocks @ H.T, per block of gs input channels."""
    out_f, in_f = w.shape
    if in_f % gs != 0:
        raise ValueError(f"in_features {in_f} not divisible by group size {gs}")
    return (w.reshape(out_f, in_f // gs, gs) @ h.T.to(w)).reshape(out_f, in_f)


def rotate_activation(x: Tensor, h: Tensor, gs: int) -> Tensor:
    """Online: x_rot = x_blocks @ H, computed in x's dtype (as comfy-kitchen)."""
    k = x.shape[-1]
    if k % gs != 0:
        raise ValueError(f"features {k} not divisible by group size {gs}")
    return (x.reshape(-1, k // gs, gs) @ h.to(x)).reshape(x.shape)


def quantize_rowwise(x: Tensor) -> tuple[Tensor, Tensor]:
    """Dynamic per-row INT8: scale = absmax/127 (f32), q = round(x/scale).
    The division runs in x's dtype, matching comfy-kitchen's quantize_int8_rowwise."""
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    s = scale.to(x.dtype)
    s = torch.where(s == 0, torch.full_like(s, torch.finfo(x.dtype).tiny), s)
    q = (x / s).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale


@torch.no_grad()
def quantize_weight_convrot(w: Tensor, gs: int | None, mseclip: bool = True) -> tuple[Tensor, Tensor]:
    """Rotate (if gs) and per-output-row quantize a weight to int8 in [-127, 127].

    mseclip: per row, search clip ratios in [0.55, 1] of absmax and keep the
    one minimizing reconstruction MSE (Comfy-Org quant_int8_convrot.py recipe).
    Returns (int8 [N, K], f32 scale [N, 1]) on w's device.
    """
    wf = w.float()
    if gs:
        wf = rotate_weight(wf, regular_hadamard(gs, wf.device), gs)
    absmax = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    if not mseclip:
        scale = (absmax / 127.0).clamp(min=1e-30)
        return (wf / scale).round().clamp(-127, 127).to(torch.int8), scale
    best_mse = torch.full_like(absmax, float("inf"))
    best_scale = absmax / 127.0
    best_q = None
    for a in CLIP_GRID.tolist():
        scale = (absmax * a / 127.0).clamp(min=1e-30)
        q = (wf / scale).round_().clamp_(-127, 127)
        mse = ((q * scale - wf) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q if best_q is None else torch.where(better, q, best_q)
    return best_q.to(torch.int8), best_scale


def dequantize_weight(q: Tensor, scale: Tensor, gs: int | None) -> Tensor:
    """int8 + scale -> f32 weight in the ORIGINAL (unrotated) basis."""
    w = q.float() * scale.float()
    return rotate_weight(w, regular_hadamard(gs, w.device), gs) if gs else w


def marker_dict(gs: int | None) -> dict:
    """The per-layer config, key order as ComfyUI's own save path writes it."""
    d = {"format": COMFY_FORMAT}
    if gs:
        d["convrot"] = True
        d["convrot_groupsize"] = gs
    return d


def comfy_quant_bytes(gs: int | None) -> Tensor:
    return torch.tensor(list(json.dumps(marker_dict(gs)).encode("utf-8")), dtype=torch.uint8)


def parse_comfy_quant(t: Tensor) -> int | None:
    """Marker tensor -> convrot group size (None = unrotated). Rejects other formats."""
    cfg = json.loads(bytes(t.cpu().tolist()).decode("utf-8"))
    if cfg.get("format") != COMFY_FORMAT:
        raise ValueError(f"unsupported comfy_quant format: {cfg}")
    return int(cfg.get("convrot_groupsize", 256)) if cfg.get("convrot") else None


def _round_up(v: int, m: int) -> int:
    return (v + m - 1) // m * m


def int8_mm(a: Tensor, b_t: Tensor) -> Tensor:
    """int32 [M, N] = a int8 [M, K] @ b_t.T, b_t int8 [N, K].

    CUDA: torch._int_mm (cuBLASLt) with M padded to >= 32 / multiple of 32 and
    N to a multiple of 8, as comfy-kitchen does. CPU: exact int32 matmul, so
    the CPU path is the same quantized math, just slow.
    """
    if not a.is_cuda:
        return a.to(torch.int32) @ b_t.to(torch.int32).T
    m, k = a.shape
    n = b_t.shape[0]
    pm, pn = _round_up(max(m, 32), 32), _round_up(n, 8)
    if pm != m:
        a = F.pad(a, (0, 0, 0, pm - m))
    if pn != n:
        b_t = F.pad(b_t, (0, 0, 0, pn - n))
    if k % 8 != 0:
        a, b_t = F.pad(a, (0, 8 - k % 8)), F.pad(b_t, (0, 8 - k % 8))
    out = torch._int_mm(a, b_t.T)
    return out[:m, :n]


class ConvRotLinearW8A8(nn.Module):
    """Drop-in nn.Linear replacement holding a ComfyUI int8_tensorwise(+ConvRot) weight.

    forward: rotate x per block (online), per-token int8 quant, int8 GEMM,
    rescale by x_scale * w_scale, add bias. `weight_scale` is kept f32 through
    `.to(dtype)` casts — a bf16 scale would cost ~0.4% per output channel.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 groupsize: int | None = 256, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.groupsize = groupsize
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device)) if bias else None

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, convrot_groupsize={self.groupsize}")

    def _apply(self, fn, recurse=True):
        scale = self._buffers.pop("weight_scale")
        try:
            super()._apply(fn, recurse)
        finally:
            self._buffers["weight_scale"] = scale.to(self.weight.device)
        return self

    @classmethod
    def from_linear(cls, lin: nn.Linear, groupsize: int | None = 256,
                    mseclip: bool = True) -> "ConvRotLinearW8A8":
        m = cls(lin.in_features, lin.out_features, lin.bias is not None, groupsize,
                device=lin.weight.device)
        q, s = quantize_weight_convrot(lin.weight.detach(), groupsize, mseclip)
        m.weight.copy_(q)
        m.weight_scale.copy_(s)
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.detach())
        return m

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "comfy_quant"] = comfy_quant_bytes(self.groupsize)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "comfy_quant"
        if key in state_dict:
            self.groupsize = parse_comfy_quant(state_dict.pop(key))
        elif strict:
            missing_keys.append(key)
        sk = prefix + "weight_scale"
        if sk in state_dict:
            state_dict[sk] = state_dict[sk].float().reshape(self.out_features, 1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def dequantized_weight(self) -> Tensor:
        return dequantize_weight(self.weight, self.weight_scale, self.groupsize)

    def forward(self, x: Tensor) -> Tensor:
        out_dtype = x.dtype
        if self.groupsize:
            x = rotate_activation(x, regular_hadamard(self.groupsize, x.device, x.dtype), self.groupsize)
        lead = x.shape[:-1]
        x8, xs = quantize_rowwise(x.reshape(-1, x.shape[-1]))
        acc = int8_mm(x8, self.weight)
        y = (acc.float() * (xs * self.weight_scale.reshape(1, -1))).to(out_dtype)
        if self.bias is not None:
            y = y + self.bias.to(out_dtype)
        return y.reshape(*lead, self.out_features)


def quantized_layer_names(keys) -> list[str]:
    return [k[: -len(MARKER_SUFFIX)] for k in keys if k.endswith(MARKER_SUFFIX)]


def is_quantized_checkpoint(path: str) -> bool:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return any(k.endswith(MARKER_SUFFIX) for k in f.keys())


def swap_quantized_linears(model: nn.Module, keys) -> int:
    """Replace every nn.Linear that has a `<name>.comfy_quant` key in `keys` by
    a ConvRotLinearW8A8 on the same device. Call BEFORE load_state_dict (meta
    model + assign=True is the intended use); the group size is read from the
    marker at load time. Returns the number of swapped layers."""
    names = quantized_layer_names(keys)
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lin = getattr(parent, child)
        if not isinstance(lin, nn.Linear):
            raise TypeError(f"{name} is {type(lin).__name__}, expected nn.Linear")
        setattr(parent, child, ConvRotLinearW8A8(
            lin.in_features, lin.out_features, lin.bias is not None, device=lin.weight.device))
    return len(names)

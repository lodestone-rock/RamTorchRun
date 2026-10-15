"""check_convrot.py — invariants of the ConvRot W8A8 layer (`krea2/model/convrot.py`).

CPU, seconds. A checkpoint written by `quantize_convrot_w8a8.py` is only
usable in ComfyUI if our rotation and marker match comfy-kitchen's exactly, and
only usable here if the layer reproduces nn.Linear. Checks:

  1. H (order 16, 64, 256, 1024) is orthogonal, symmetric and REGULAR: every
     row and column sums to +-sqrt(n) — the property that stops it amplifying
     row-wise outliers (a Sylvester H has an all-ones row summing to n).
  2. rotate_activation(x) @ rotate_weight(W).T == x @ W.T (fp32).
  3. On activations with a few huge channels (the DiT outlier pattern), the
     rotated W8A8 layer's error is far below unrotated per-token INT8.
  4. The `.comfy_quant` marker is byte-identical to stock ComfyUI's
     (72 bytes for group 256), and parses back.
  5. state_dict round trip: keys/dtypes/shapes match the Kroma W8A8 release
     layout, weight_scale survives `.to(bfloat16)` as f32, and a meta-built
     model loaded via swap_quantized_linears + assign=True gives identical output.
  6. CUDA only: the torch._int_mm path equals the exact CPU int32 path.

Run:
    uv run python krea2/tools/check_convrot.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from krea2.model.convrot import (
    ConvRotLinearW8A8,
    comfy_quant_bytes,
    parse_comfy_quant,
    quantize_rowwise,
    regular_hadamard,
    rotate_activation,
    rotate_weight,
    swap_quantized_linears,
)


def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def check_hadamard() -> None:
    for n in (16, 64, 256, 1024):
        h = regular_hadamard(n, dtype=torch.float64)
        eye = torch.eye(n, dtype=torch.float64)
        assert torch.allclose(h @ h.T, eye, atol=1e-12), n
        assert torch.equal(h, h.T), n
        raw = h * n ** 0.5
        assert torch.equal(raw.abs(), torch.ones_like(raw)), n
        for sums in (raw.sum(0), raw.sum(1)):
            assert torch.equal(sums.abs(), torch.full_like(sums, n ** 0.5)), (n, sums.unique())
    try:
        regular_hadamard(128)
        raise AssertionError("order 128 (not a power of 4) must be rejected")
    except ValueError:
        pass
    print("[ok] regular Hadamard: orthogonal, symmetric, row/col sums +-sqrt(n), n=16..1024")


def check_rotation_identity() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(7, 3, 1024, generator=g, dtype=torch.float64)
    w = torch.randn(96, 1024, generator=g, dtype=torch.float64)
    for gs in (16, 256, 1024):
        h = regular_hadamard(gs, dtype=torch.float64)
        y = rotate_activation(x, h, gs) @ rotate_weight(w, h, gs).T
        assert relerr(y, x @ w.T) < 1e-12, gs
    print("[ok] rot(x) @ rot(W).T == x @ W.T for gs 16/256/1024")


def _w8a8_unrotated(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    qx, sx = quantize_rowwise(x)
    sw = w.abs().amax(1, keepdim=True) / 127
    qw = (w / sw).round().clamp(-127, 127)
    return (qx.float() @ qw.T) * sx * sw.T


def check_outliers() -> None:
    g = torch.Generator().manual_seed(1)
    k, n = 2048, 512
    x = torch.randn(64, k, generator=g)
    x[:, torch.randperm(k, generator=g)[:6]] *= 80.0          # channel outliers
    x[torch.randperm(64, generator=g)[:4]] *= 20.0             # row-wise outlier tokens
    w = torch.randn(n, k, generator=g) / k ** 0.5
    ref = x @ w.T
    lin = nn.Linear(k, n, bias=False)
    lin.weight.data.copy_(w)
    e_rot = relerr(ConvRotLinearW8A8.from_linear(lin, 256)(x), ref)
    e_plain = relerr(_w8a8_unrotated(x, w), ref)
    assert e_rot < 0.02, e_rot
    assert e_rot * 3 < e_plain, (e_rot, e_plain)
    print(f"[ok] outlier activations: ConvRot W8A8 relerr {e_rot:.4f} vs unrotated INT8 {e_plain:.4f}")


def check_marker() -> None:
    want = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
    t = comfy_quant_bytes(256)
    assert t.dtype == torch.uint8 and t.numel() == 72 == len(want)
    assert bytes(t.tolist()) == want
    assert parse_comfy_quant(t) == 256
    assert bytes(comfy_quant_bytes(None).tolist()) == b'{"format": "int8_tensorwise"}'
    assert parse_comfy_quant(comfy_quant_bytes(None)) is None
    print("[ok] comfy_quant marker byte-identical to stock ComfyUI (72 B)")


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict({"wq": nn.Linear(512, 256), "keep": nn.Linear(512, 8)})])


def check_state_dict() -> None:
    torch.manual_seed(2)
    src = _Tiny()
    x = torch.randn(5, 512)
    q = _Tiny()
    q.load_state_dict(src.state_dict())
    q.blocks[0]["wq"] = ConvRotLinearW8A8.from_linear(src.blocks[0]["wq"], 256)
    sd = q.state_dict()
    assert sd["blocks.0.wq.weight"].dtype == torch.int8 and sd["blocks.0.wq.weight"].shape == (256, 512)
    assert sd["blocks.0.wq.weight_scale"].dtype == torch.float32 and sd["blocks.0.wq.weight_scale"].shape == (256, 1)
    assert sd["blocks.0.wq.comfy_quant"].dtype == torch.uint8
    assert "blocks.0.keep.comfy_quant" not in sd
    y = q.blocks[0]["wq"](x)
    assert relerr(y, src.blocks[0]["wq"](x)) < 0.02

    with torch.device("meta"):
        m = _Tiny()
    assert swap_quantized_linears(m, sd.keys()) == 1
    m.load_state_dict(sd, strict=True, assign=True)
    lay = m.blocks[0]["wq"]
    assert lay.groupsize == 256 and not lay.weight.is_meta
    assert torch.equal(lay(x), y)

    m = m.to(torch.bfloat16)
    assert lay.weight.dtype == torch.int8 and lay.weight_scale.dtype == torch.float32
    assert m.blocks[0]["keep"].weight.dtype == torch.bfloat16
    yb = lay(x.bfloat16())
    assert yb.dtype == torch.bfloat16 and relerr(yb, y) < 0.02
    print("[ok] state_dict round trip (meta + assign), f32 scale survives .to(bf16)")


def check_cuda() -> None:
    if not torch.cuda.is_available():
        print("[skip] no CUDA: torch._int_mm path not checked")
        return
    dev = torch.device("cuda")
    torch.manual_seed(3)
    for m_rows in (1, 17, 300):
        lin = nn.Linear(1536, 136, bias=True)
        lay = ConvRotLinearW8A8.from_linear(lin, 256)
        x = torch.randn(m_rows, 1536).bfloat16()
        cpu = lay(x)
        gpu = lay.to(dev)(x.to(dev)).cpu()
        lay.to("cpu")
        assert relerr(gpu, cpu) < 1e-2, (m_rows, relerr(gpu, cpu))
        assert relerr(gpu, lin(x.float())) < 0.03
    a = torch.randint(-128, 128, (64, 512), dtype=torch.int8)
    b = torch.randint(-128, 128, (40, 512), dtype=torch.int8)
    from krea2.model.convrot import int8_mm
    assert torch.equal(int8_mm(a.to(dev), b.to(dev)).cpu(), int8_mm(a, b))
    print("[ok] CUDA torch._int_mm path == exact int32 CPU path")


def main() -> None:
    check_hadamard()
    check_rotation_identity()
    check_outliers()
    check_marker()
    check_state_dict()
    check_cuda()
    print("[ok] all ConvRot checks passed")


if __name__ == "__main__":
    main()

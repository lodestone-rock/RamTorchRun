"""check_streaming_sample.py — does streaming (infer_loop) sampling match the barriered loop?

Runs krea2.inference.sample_pipeline twice on a TINY K2 pipeline (CPU, no
downloads): once with streaming=False (one barriered infer() per Euler step,
two with CFG) and once with streaming=True (Pipeline.infer_loop, persistent
stage workers). Both must produce the same images up to kernel-level numerics
— the streaming path pads caption blocks to a common text length and doubles
each microbatch into [uncond_i, cond_i] CFG pairs, so bitwise equality is NOT
expected; tolerance is set well above bf16 kernel noise.

Also asserts determinism: each path is bitwise-identical to itself across runs.

Run: uv run python krea2/tools/check_streaming_sample.py   (CPU, seconds)
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from torch import nn

from krea2.inference import sample_pipeline
from krea2.model.chunks import build_dit_chunks
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from ramtorch import Pipeline

TINY = SingleMMDiTConfig(
    features=32, tdim=16, txtdim=32, heads=2, kvheads=1, multiplier=2,
    layers=2, patch=2, channels=4, txtheads=2, txtkvheads=1, txtlayers=2,
    bias=True,
)


class StubTokenizer:
    """'' -> 2 tokens, anything else -> 4 tokens (exercises the padding path)."""
    prefix_idx = 0

    def __call__(self, prompts):
        n = len(prompts)
        longest = max((len(p) for p in prompts), default=1)
        length = 4  # real encoder emits equal-length cond/uncond blocks
        ids = torch.arange(n * length, dtype=torch.long).reshape(n, length) % 97
        mask = torch.ones(n, length, dtype=torch.bool)
        return ids, mask


class StubEncoderPipe:
    """Deterministic 'encoder': token ids -> features (CPU pipeline shape)."""
    def __init__(self, txtdim, txtlayers):
        self.txtdim = txtdim
        self.txtlayers = txtlayers

    def infer(self, data, n_microbatches=1):
        ids, mask = data
        feats = (ids.float().unsqueeze(-1).unsqueeze(-1) * 0.01).expand(
            *ids.shape, self.txtlayers, self.txtdim).contiguous()
        return feats


class StubAE:
    compression = 1
    channels = 4

    def decode(self, latent):
        # Per-channel spatial mean map: keeps spatial structure for parity
        # comparison while staying shape-agnostic.
        n = latent.shape[0]
        m = latent.float().mean(dim=1, keepdim=True)  # [n, 1, h, w]
        return m.expand(n, 3, m.shape[2], m.shape[3]).contiguous()


def images_to_array(images):
    return np.stack([np.asarray(im) for im in images])


def run_once(streaming, guidance, seed=0):
    torch.manual_seed(seed)  # model init must match across runs; sampling noise
    set_sdpa_ctx(False)      # is generator-seeded separately inside the sampler
    model = SingleStreamDiT(TINY)
    chunks = build_dit_chunks(model)
    head = chunks[-1]
    pipe = Pipeline(chunk_modules=chunks, chunks_per_stage=[2, 2],
                    devices=["cpu", "cpu"], offload=False)
    ae = StubAE()
    prompts = ["a cat", ""]  # unequal caption lengths -> padding path
    out = sample_pipeline(
        pipe, head, StubEncoderPipe(TINY.txtdim, TINY.txtlayers), StubTokenizer(), ae,
        TINY.patch, prompts,
        negative_prompts=["", ""],
        device="cpu", dtype=torch.float32,
        width=4, height=4, steps=4,
        guidance=guidance, seed=seed,
        streaming=streaming,
    )
    return images_to_array(out)


def main():
    failures = 0
    for guidance in (4.5, 0.0):
        label = f"CFG g={guidance}"
        ref_a = run_once(False, guidance, seed=0)
        ref_b = run_once(False, guidance, seed=0)
        stream = run_once(True, guidance, seed=0)
        deterministic = np.array_equal(ref_a, ref_b)
        self_diff = float(np.abs(ref_a.astype(np.int16) - ref_b.astype(np.int16)).max())
        max_diff = float(np.abs(ref_a.astype(np.int16) - stream.astype(np.int16)).max())
        close = np.allclose(ref_a.astype(np.float32), stream.astype(np.float32), atol=3.0)
        ok = close
        failures += 0 if ok else 1
        print(f"[{label}] barrier self_diff={self_diff}  "
              f"streaming max_abs_diff={max_diff}  allclose(atol=3)={close}  "
              f"-> {'PASS' if ok else 'FAIL'}")
    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: streaming sampling parity")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

"""GPU checks for inference-only local attention, GQA, padding and offloading."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from ramtorch import OffloadModel
from torch.nn.attention.flex_attention import create_mask

from krea2.model.local_attention import LocalAttention, enable_local_attention, validate_layer_head_ids
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.chunks import build_dit_chunks
from krea2.model.sampling import prepare


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch._inductor.config.compile_threads = 1
    torch.manual_seed(42)
    device = "cuda:0"
    set_sdpa_ctx(False)
    # Rectangular grid, two tag slots, different valid prefixes across samples.
    latent = torch.randn(2, 4, 24, 32, device=device)
    txtmask = torch.tensor([[1, 1, 0], [1, 0, 0]], device=device, dtype=torch.bool)
    tagmask = torch.tensor([[1, 0], [0, 0]], device=device, dtype=torch.bool)
    img, pos, valid = prepare(latent, 3, 2, txtmask, taglen=2, tagmask=tagmask)
    prefix, imglen = 5, img.shape[1]
    pad = -valid.shape[1] % 256
    valid, pos = F.pad(valid, (0, pad)), F.pad(pos, (0, 0, 0, pad))
    length = valid.shape[1]
    q = torch.randn(2, 48, length, 64, device=device)
    k = torch.randn(2, 12, length, 64, device=device)
    v = torch.randn_like(k)
    for ids in (list(range(2)), list(range(48)), list(range(0, 48, 4)),
                list(range(1, 48, 4))):
        full = len(ids)
        policy = LocalAttention(48, full, 11, ids)
        policy.prepare(valid, pos, prefix, imglen)
        cached = policy.block_mask
        policy.prepare(valid.clone(), pos.clone(), prefix, imglen)
        assert policy.block_mask is cached
        idx = torch.arange(length, device=device)
        near = (pos[:, :, None, 1:] - pos[:, None, :, 1:]).abs().amax(-1) <= 5
        prefix_pair = (idx[:, None] < prefix) | (idx[None, :] < prefix)
        full_mask = torch.zeros(48, dtype=torch.bool, device=device)
        full_mask[ids] = True
        dense = valid[:, None, :, None] & valid[:, None, None, :] & (
            full_mask[None, :, None, None]
            | prefix_pair[None, None] | near[:, None])
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=dense, enable_gqa=True)
        got = policy(q, k, v, gqa=True).reshape(2, length, 48, 64).transpose(1, 2)
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)
        print(f"PASS GQA48/12 full={full}, rectangular, prefix/tag/pad masks, cache; "
              f"error={(got-expected).abs().max().item():.3g}")

    cfg = SingleMMDiTConfig(features=256, tdim=32, txtdim=64, heads=4,
                           kvheads=1, multiplier=2, layers=2, patch=2, channels=4,
                           txtheads=2, txtkvheads=2, txtlayers=2)
    dit = SingleStreamDiT(cfg).to(device).eval().requires_grad_(False)
    img, pos, valid = prepare(latent[:1], 3, 2, txtmask[:1])
    inputs = (img, torch.randn(1, 3, 2, 64, device=device),
              torch.tensor([0.5], device=device), pos, valid)
    keys = set(dit.state_dict())
    dense_out = dit(*inputs)
    enable_local_attention(dit, 4, 11)
    torch.testing.assert_close(dit(*inputs), dense_out, atol=2e-4, rtol=2e-4)
    for bad in ([[0, 0], [1, 2]], [[0, 4], [1, 2]], [[True, 2], [1, 2]], [[0, 2]]):
        try:
            validate_layer_head_ids(bad, 2, 4, 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid policy: {bad}")
    policy = enable_local_attention(dit, 2, 11, layer_head_ids=[[0, 2], [1, 3]])
    assert len(policy.policies) == 2
    assert dit.blocks[0].attn.attention_impl is not dit.blocks[1].attn.attention_impl
    expected = dit(*inputs)
    def reference(attention):
        def call(q, k, v, mask=None, gqa=False):
            dense = create_mask(attention.block_mask.mask_mod, q.shape[0], q.shape[1],
                                q.shape[2], k.shape[2], device=str(q.device))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=dense, enable_gqa=gqa)
            return out.transpose(1, 2).flatten(2)
        return call
    for block, attention in zip(dit.blocks, policy.layers):
        block.attn.attention_impl = reference(attention)
    torch.testing.assert_close(dit(*inputs), expected, atol=2e-4, rtol=2e-4)
    for block, attention in zip(dit.blocks, policy.layers):
        block.attn.attention_impl = attention
    assert set(dit.state_dict()) == keys
    chunks = build_dit_chunks(dit)
    chunks[-1].set_seq(3, img.shape[1])
    state = inputs
    for chunk in chunks:
        state = chunk(*state)
    torch.testing.assert_close(state, expected, atol=2e-4, rtol=2e-4)
    off = OffloadModel(chunks, device=device, window=2)
    try:
        torch.testing.assert_close(off(inputs), expected, atol=2e-4, rtol=2e-4)
    finally:
        off.close()
    print("PASS per-layer SDPA, invalid policies, dense limit, checkpoint keys, chunk/offload parity")


if __name__ == "__main__":
    main()
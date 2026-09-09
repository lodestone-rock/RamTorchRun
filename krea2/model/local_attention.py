"""Opt-in, uncalibrated joint-DiT local attention for single-GPU inference.

Plain Python policies (not nn.Modules): no checkpoint or streamed parameter state.
The embed prepares cached BlockMasks before the block stack starts executing.
"""
from collections import OrderedDict

import torch
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention


class LocalAttention:
    def __init__(self, heads: int, full_heads: int = 2, window: int = 11,
                 full_head_ids=None):
        if not 1 <= full_heads <= heads:
            raise ValueError("full_heads must be between 1 and the number of query heads")
        if window < 1 or window % 2 != 1:
            raise ValueError("window must be a positive odd integer")
        self.heads, self.full_heads, self.window = heads, full_heads, window
        self.full_head_ids = tuple(range(full_heads) if full_head_ids is None else full_head_ids)
        if (len(self.full_head_ids) != full_heads or
                len(set(self.full_head_ids)) != full_heads or
                any(not isinstance(h, int) or not 0 <= h < heads for h in self.full_head_ids)):
            raise ValueError("full_head_ids must contain full_heads unique valid query-head indices")
        self.cache = OrderedDict()
        self.block_mask = None
        self.build = torch.compile(create_block_mask, fullgraph=True, dynamic=False)
        self.flex = torch.compile(flex_attention, fullgraph=True, dynamic=False)

    def prepare(self, valid, pos, prefix, imglen):
        if torch.is_grad_enabled():
            raise RuntimeError("LocalAttention is an inference-only experiment; use torch.no_grad()")
        if valid.device.type != "cuda":
            raise ValueError("LocalAttention requires CUDA")
        if valid.shape[1] % 128:
            raise ValueError("LocalAttention expects the DiT's block-aligned padded sequence")
        # Small host key, not an SxS mask. Includes CFG validity, rectangular
        # geometry, tag slots and padding. Keep clones alive through mask_mod.
        coordinates = pos[..., 1:].to(torch.int32)
        key = (str(valid.device), prefix, imglen, tuple(valid.shape),
               valid.cpu().numpy().tobytes(), coordinates.cpu().numpy().tobytes())
        if key in self.cache:
            self.block_mask = self.cache[key]
            self.cache.move_to_end(key)
            return
        valid = valid.detach().clone()
        ys = pos[..., 1].to(torch.int32).contiguous()
        xs = pos[..., 2].to(torch.int32).contiguous()
        radius = self.window // 2
        full = torch.zeros(self.heads, dtype=torch.bool, device=valid.device)
        full[list(self.full_head_ids)] = True

        def mask_mod(b, h, q, kv):
            near = ((ys[b, q] - ys[b, kv]).abs() <= radius) & (
                (xs[b, q] - xs[b, kv]).abs() <= radius)
            return valid[b, q] & valid[b, kv] & (
                full[h] | (q < prefix) | (kv < prefix) | near)

        # Construct only two unique head lists, then expand the sparse metadata.
        def two_heads(b, h, q, kv):
            near = ((ys[b, q] - ys[b, kv]).abs() <= radius) & (
                (xs[b, q] - xs[b, kv]).abs() <= radius)
            return valid[b, q] & valid[b, kv] & (
                (h == 0) | (q < prefix) | (kv < prefix) | near)

        length = valid.shape[1]
        two = self.build(two_heads, valid.shape[0], 2, length, length,
                         device=str(valid.device), BLOCK_SIZE=128)
        selector = torch.ones(self.heads, dtype=torch.long, device=valid.device)
        selector[list(self.full_head_ids)] = 0
        bm = BlockMask.from_kv_blocks(
            two.kv_num_blocks.index_select(1, selector),
            two.kv_indices.index_select(1, selector),
            two.full_kv_num_blocks.index_select(1, selector),
            two.full_kv_indices.index_select(1, selector),
            BLOCK_SIZE=128, mask_mod=mask_mod, seq_lengths=(length, length),
        )
        self.cache[key] = self.block_mask = bm
        if len(self.cache) > 4:
            self.cache.popitem(last=False)
        count = (bm.kv_num_blocks.sum() + bm.full_kv_num_blocks.sum()).item()
        total = valid.shape[0] * self.heads * (length // 128)**2
        print(f"[local-attn] prefix={prefix}, image={imglen}, padded={length}, "
              f"full={self.full_heads}/{self.heads}, window={self.window}, "
              f"active blocks={100 * count / total:.2f}%", flush=True)

    def __call__(self, q, k, v, mask=None, gqa=False):
        if torch.is_grad_enabled() or self.block_mask is None:
            raise RuntimeError("Prepare LocalAttention under no_grad before inference")
        out = self.flex(q, k, v, block_mask=self.block_mask, enable_gqa=gqa)
        return rearrange(out, "B H L D -> B L (H D)")


def validate_layer_head_ids(layer_head_ids, layers, heads, full_heads):
    """Validate a JSON-compatible layer-by-head selection before loading weights."""
    if not isinstance(layer_head_ids, list) or len(layer_head_ids) != layers:
        raise ValueError(f"Expected one head list for each of {layers} layers")
    for ids in layer_head_ids:
        if (not isinstance(ids, list) or len(ids) != full_heads or
                any(type(h) is not int or not 0 <= h < heads for h in ids) or
                len(set(ids)) != full_heads):
            raise ValueError(f"Each layer needs {full_heads} unique integer heads in [0,{heads})")
    return [list(ids) for ids in layer_head_ids]


class LayerLocalAttention:
    """Prepare each distinct head policy once; blocks hold immutable policy refs.

    Only used for single-GPU inference, like LocalAttention. No mutable layer
    counter: checkpointing/offload execution order cannot select the wrong mask.
    """

    def __init__(self, heads, full_heads, window, layer_head_ids):
        unique = {}
        self.layers = []
        for ids in layer_head_ids:
            key = tuple(sorted(ids))
            if key not in unique:
                unique[key] = LocalAttention(heads, full_heads, window, key)
            self.layers.append(unique[key])
        self.policies = tuple(unique.values())

    def prepare(self, valid, pos, prefix, imglen):
        for policy in self.policies:
            policy.prepare(valid, pos, prefix, imglen)


def enable_local_attention(dit, full_heads=2, window=11, full_head_ids=None,
                           layer_head_ids=None):
    if layer_head_ids is not None:
        if full_head_ids is not None:
            raise ValueError("Use either a global head list or per-layer lists, not both")
        ids = validate_layer_head_ids(layer_head_ids, len(dit.blocks),
                                      dit.config.heads, full_heads)
        policy = LayerLocalAttention(dit.config.heads, full_heads, window, ids)
        dit.local_attention = policy
        for block, attention in zip(dit.blocks, policy.layers):
            block.attn.attention_impl = attention
        return policy
    policy = LocalAttention(dit.config.heads, full_heads, window, full_head_ids)
    dit.local_attention = policy
    for block in dit.blocks:
        block.attn.attention_impl = policy
    return policy
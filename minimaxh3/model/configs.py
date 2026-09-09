"""configs.py — MiniMax-H3 model configurations.

The DiT config mirrors the diffusers `MiniMaxH3Transformer3DModel` config
(``checkpoints/minimaxh3/transformer/config.json``) field for field, so the
ported `MiniMaxH3DiT` builds a module tree whose state dict matches the
official checkpoint exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class H3DiTConfig:
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @classmethod
    def from_json(cls, path: str) -> "H3DiTConfig":
        with open(path) as f:
            data = json.load(f)
        data.pop("_class_name", None)
        data.pop("_diffusers_version", None)
        if "patch_size" in data:
            data["patch_size"] = tuple(data["patch_size"])
        return cls(**data)


H3_CONFIGS = {
    "h3-33b": H3DiTConfig(),
}

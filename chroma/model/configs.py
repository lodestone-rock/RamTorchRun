"""Named Chroma model configurations."""
from .encoder import TextEncoderConfig
from .model import ChromaParams

CHROMA_CONFIGS = {
    # Chroma1 (Chroma1-HD / Chroma1-Base): ~8.9B, 19 double + 38 single blocks.
    "chroma1": ChromaParams(
        in_channels=64,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        approximator_in_dim=64,
        approximator_depth=5,
        approximator_hidden_size=5120,
    ),
    # Tiny CPU config for the parity/role tools (~1M params, structurally
    # faithful: pe_dim = hidden/heads = 32 = sum(axes_dim)).
    "tiny": ChromaParams(
        in_channels=16,           # patch 2 x 2 of 4 latent channels
        context_in_dim=32,
        hidden_size=64,
        mlp_ratio=2.0,
        num_heads=2,
        depth=2,
        depth_single_blocks=3,
        axes_dim=[8, 12, 12],
        theta=10_000,
        qkv_bias=True,
        approximator_in_dim=32,
        approximator_depth=2,
        approximator_hidden_size=64,
        channels=4,
    ),
}

ENCODER_CONFIGS = {
    # Chroma conditions on the raw T5-XXL encoder output (last hidden state).
    "t5_xxl": TextEncoderConfig(
        model_id="lodestones/Chroma1-HD",
        subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        max_length=512,
    ),
}

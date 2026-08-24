"""Named Radiance model configurations."""
from .encoder import TextEncoderConfig
from .model import RadianceParams

RADIANCE_CONFIGS = {
    # Radiance x0 patch-16: ~9.5B, 19 double + 38 single blocks + NeRF head.
    # 659 tensors, matching /mnt/datapool_u2/models/radiance/latest_x0.safetensors.
    "radiance_x0_p16": RadianceParams(
        in_channels=3,
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
        nerf_hidden_size=64,
        nerf_mlp_ratio=4,
        nerf_depth=4,
        nerf_max_freqs=8,
        patch_size=16,
    ),
    # Tiny CPU config for the parity/role tools (~1M params, structurally
    # faithful: pe_dim = hidden/heads = 32 = sum(axes_dim)). patch_size 4 with
    # 8x8 images gives 4 patches per sample, enough to catch a wrong fold.
    "tiny": RadianceParams(
        in_channels=3,
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
        nerf_hidden_size=8,
        nerf_mlp_ratio=2,
        nerf_depth=2,
        nerf_max_freqs=4,
        patch_size=4,
    ),
}

ENCODER_CONFIGS = {
    # Same T5-XXL as Chroma (already cached from the chroma runs).
    "t5_xxl": TextEncoderConfig(
        model_id="lodestones/Chroma1-HD",
        subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        max_length=512,
    ),
    # The copy that ships next to the radiance weights — same tensors, but
    # under diffusers' *_2 subfolder names. Use this to run without HF network.
    "t5_xxl_local": TextEncoderConfig(
        model_id="/mnt/datapool_u2/models/radiance/chroma",
        subfolder="text_encoder_2",
        tokenizer_subfolder="tokenizer_2",
        max_length=512,
    ),
}

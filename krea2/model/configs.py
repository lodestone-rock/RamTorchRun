"""Named K2 model configurations.

Extracted from x0-pred's krea2_trainer.py so that trainer and inference
scripts can share them without importing the full training module.
"""
from .encoder import TextEncoderConfig
from .mmdit import SingleMMDiTConfig

MMDIT_CONFIGS = {
    "large_wide": SingleMMDiTConfig(
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        txtheads=20,
        txtkvheads=20,
        txtlayers=12,
    ),
}

ENCODER_CONFIGS = {
    "qwen3_vl_4b": TextEncoderConfig(model_id="Qwen/Qwen3-VL-4B-Instruct"),
}

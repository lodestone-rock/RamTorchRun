"""Radiance model package: pixel-space rectified-flow transformer with a NeRF
decoder head, T5-XXL text encoder, LoRA utilities, sampling, and RamTorch chunk
dicing. There is no autoencoder — the model reads and writes pixels."""
from .configs import ENCODER_CONFIGS, RADIANCE_CONFIGS
from .encoder import T5Conditioner, TextEncoderConfig
from .model import Radiance, RadianceParams
from .sampling import prepare_image_ids, sample, timesteps

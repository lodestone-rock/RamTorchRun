"""Chroma model package: rectified-flow transformer, Flux VAE, T5-XXL text
encoder, LoRA utilities, sampling, and RamTorch chunk dicing."""
from .autoencoder import FluxAutoencoder
from .configs import CHROMA_CONFIGS, ENCODER_CONFIGS
from .encoder import T5Conditioner, TextEncoderConfig
from .model import Chroma, ChromaParams
from .sampling import prepare, sample, timesteps

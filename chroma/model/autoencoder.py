import torch
from torch import Tensor, nn


class FluxAutoencoder(nn.Module):
    """The Flux VAE (f8, 16 latent channels) as shipped in the Chroma1-HD repo."""

    def __init__(self, model_id: str = "lodestones/Chroma1-HD", subfolder: str = "vae"):
        super().__init__()
        from diffusers import AutoencoderKL

        self.ae = AutoencoderKL.from_pretrained(model_id, subfolder=subfolder)
        self.compression = 8
        self.channels = self.ae.config.latent_channels          # 16
        self.scaling = float(self.ae.config.scaling_factor)     # 0.3611
        self.shift = float(self.ae.config.shift_factor)         # 0.1159

    def decode(self, x: Tensor) -> Tensor:
        # Inverse of the encode normalization: z/scale + shift.
        return self.ae.decode(x / self.scaling + self.shift).sample

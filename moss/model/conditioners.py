"""MOSS-Image conditioners: Qwen3.5 text encoder (frozen) and the FLUX.2 VAE decoder (fp32, as in the release)."""
from __future__ import annotations

from pathlib import Path

import torch


class TextEncoder:
    """Prompt -> unpadded last hidden state [L, 4096]. Prompts run one at a time, like the release:
    Qwen3.5's gated-delta layers are not padding-invariant."""

    def __init__(self, path, device, max_length=4096):
        from transformers import AutoModel, AutoTokenizer
        path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer" if (path / "tokenizer").is_dir() else path)
        self.model = AutoModel.from_pretrained(path / "text_encoder", dtype=torch.bfloat16).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.max_length = device, max_length

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> list[torch.Tensor]:
        rows = []
        for prompt in prompts:
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=True)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
            output = self.model(**inputs.to(self.device), use_cache=False, return_dict=True)
            rows.append(output.last_hidden_state[0])
        return rows

    def token_count(self, prompt: str) -> int:
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        return len(self.tokenizer(text)["input_ids"])


def patchify_latent(z: torch.Tensor) -> torch.Tensor:
    """VAE latent [B, c, 2h, 2w] -> model layout [B, 4c, h, w]; the inverse of ``unpatchify_latent``."""
    b, c, hh, ww = z.shape
    return z.reshape(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, 4 * c, hh // 2, ww // 2)


def unpatchify_latent(z: torch.Tensor) -> torch.Tensor:
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


class Encoder:
    """[-1, 1] pixels -> model latent; the exact inverse of ``Decoder`` (posterior mean, 2x2 patchify, BN)."""

    def __init__(self, path, device, vae=None):
        if vae is None:
            from diffusers import AutoencoderKLFlux2
            vae = AutoencoderKLFlux2.from_pretrained(Path(path) / "vae", torch_dtype=torch.float32).to(device).eval()
        self.vae = vae.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        vae = self.vae
        z = patchify_latent(vae.encode(pixels.to(vae.device, vae.dtype)).latent_dist.mean)
        bn, eps = vae.bn, vae.config.batch_norm_eps
        return ((z - bn.running_mean.to(z).view(1, -1, 1, 1)) / (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1)).float()


class Decoder:
    def __init__(self, path, device, vae=None):
        if vae is None:
            from diffusers import AutoencoderKLFlux2
            vae = AutoencoderKLFlux2.from_pretrained(Path(path) / "vae", torch_dtype=torch.float32).to(device).eval()
        self.vae = vae.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, 128, h, w] model latent -> [B, 3, 16h, 16w] in [-1, 1]."""
        vae = self.vae
        z = latent.to(vae.dtype)
        bn, eps = vae.bn, vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
        return vae.decode(unpatchify_latent(z), return_dict=False)[0].clamp(-1, 1)


def to_pil(images: torch.Tensor):
    from PIL import Image
    array = ((images.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in array]

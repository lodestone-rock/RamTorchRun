# Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from diffusers (`AutoencoderKLMiniMaxH3Audio`) to plain PyTorch:
# no diffusers imports, same module tree and numerics, so the diffusers-format
# safetensors checkpoint loads with `strict=True`.

"""MiniMax-H3 audio autoencoder.

Waveform in / waveform out — there is no mel front-end and no separate vocoder:

* the **encoder** is a DAC-lineage strided convolutional stack (Snake activations, weight-normed `Conv1d`) that
  downsamples by `prod(encoder_rates) = 800`, i.e. 40 latents/s at 32 kHz;
* a **causal-attention projection** (`pre_block`) rewires the 2048-wide encoder trunk to the 32-channel latent width,
  followed by the `mean_proj` / `logs_proj` posterior heads;
* the **decoder** is BigVGAN (anti-aliased SnakeBeta activations, transposed-conv upsamplers, AMP residual blocks)
  preceded by `dec_in_proj`, upsampling by `prod(decoder_rates) = 800`.

The autoencoder is **mono**. MiniMax-H3 carries stereo as two *batch* items — the caller decodes `[2, 32, T]` into
`[2, 1, samples]` and interleaves at the output boundary — so no stereo handling belongs here.

Latents are normalized with per-channel `latents_mean` / `latents_std` (32 floats each) rather than a scalar
`scaling_factor`; both live in the config and are applied by the caller.

Module and parameter names are identical to the original checkpoint, so conversion is a passthrough. That includes
`torch.nn.utils.weight_norm` (the `weight_g` / `weight_v` spelling) and the registered Kaiser-window resampling
`filter` buffers of the anti-aliased activations.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import weight_norm


class MiniMaxH3AudioDiagonalGaussianDistribution:
    r"""Posterior of the MiniMax-H3 audio autoencoder, parameterized as `(mean, log_std)`.

    The checkpoint keeps two separate `Conv1d` heads (`mean_proj`, `logs_proj`) instead of one fused moments
    projection, and the second head predicts the **log standard deviation**, not the log variance. The two tensors are
    therefore stored as produced, and `mode()` is bit-for-bit `mean_proj`'s output.

    Args:
        mean: Posterior mean, `[batch_size, latent_channels, num_frames]`.
        logs: Posterior log standard deviation, same shape as `mean`.
    """

    def __init__(self, mean: Tensor, logs: Tensor):
        self.mean = mean
        self.logs = logs
        self.std = torch.exp(logs)

    def mode(self) -> Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        if generator is not None and generator.device != self.mean.device:
            noise = torch.randn(
                self.mean.shape, generator=generator, device=generator.device, dtype=self.mean.dtype
            ).to(self.mean.device)
        else:
            noise = torch.randn(self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * noise


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> Tensor:
    r"""Kaiser-windowed sinc low-pass filter of shape `[1, 1, kernel_size]`.

    Kept arithmetically identical to the `alias-free-torch` implementation the checkpoint was trained with, because the
    resulting tensor is stored as a persistent buffer.
    """
    half_size = kernel_size // 2

    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    # Normalize to sum 1 so a constant input does not leak through the resampler.
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class MiniMaxH3AudioSnake1d(nn.Module):
    r"""`x + (alpha + 1e-9)^-1 * sin(alpha * x)^2` over `[batch_size, channels, length]`, with a
    per-channel learnable `alpha` of shape `[1, channels, 1]`. Used throughout the DAC encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)


class MiniMaxH3AudioSnakeBeta(nn.Module):
    r"""`x + (exp(beta) + 1e-9)^-1 * sin(exp(alpha) * x)^2` over `[batch_size, channels, length]`.

    The BigVGAN decoder's activation: separate frequency (`alpha`) and magnitude (`beta`) parameters, both stored in
    log space as `[channels]` vectors.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: Tensor) -> Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)


class MiniMaxH3AudioLowPassFilter1d(nn.Module):
    r"""Depthwise Kaiser-sinc low-pass filter with a stride, i.e. the anti-aliased downsampler."""

    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            hidden_states, self.filter.expand(num_channels, -1, -1), stride=self.stride, groups=num_channels
        )


class MiniMaxH3AudioUpSample1d(nn.Module):
    r"""Anti-aliased `ratio`x upsampler (transposed depthwise Kaiser-sinc convolution)."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, self.filter.expand(num_channels, -1, -1), stride=self.stride, groups=num_channels
        )
        return hidden_states[..., self.pad_left : -self.pad_right]


class MiniMaxH3AudioDownSample1d(nn.Module):
    r"""Anti-aliased `ratio`x downsampler."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = MiniMaxH3AudioLowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=kernel_size
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.lowpass(hidden_states)


class MiniMaxH3AudioActivation1d(nn.Module):
    r"""Upsample -> activation -> downsample: the alias-free activation wrapper used by BigVGAN."""

    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.downsample(hidden_states)


class MiniMaxH3AudioResidualUnit(nn.Module):
    r"""DAC residual unit: `Snake -> dilated Conv1d(k=7) -> Snake -> Conv1d(k=1)`, plus a shortcut
    that is center-cropped when the dilated convolution shrinks the time axis."""

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioSnake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=((7 - 1) * dilation) // 2)),
            MiniMaxH3AudioSnake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = self.block(hidden_states)
        pad = (hidden_states.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            hidden_states = hidden_states[..., pad:-pad]
        return hidden_states + residual


class MiniMaxH3AudioEncoderBlock(nn.Module):
    r"""Three residual units at dilations 1/3/9, then a strided channel-doubling convolution."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=1),
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=3),
            MiniMaxH3AudioResidualUnit(dim // 2, dilation=9),
            MiniMaxH3AudioSnake1d(dim // 2),
            weight_norm(
                nn.Conv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.block(hidden_states)


class MiniMaxH3AudioEncoder(nn.Module):
    r"""DAC waveform encoder: `[batch_size, 1, samples] -> [batch_size, latent_dim, samples / 800]`."""

    def __init__(self, d_model: int, strides: tuple[int, ...], d_latent: int):
        super().__init__()
        block: list[nn.Module] = [weight_norm(nn.Conv1d(1, d_model, kernel_size=7, padding=3))]
        for stride in strides:
            d_model *= 2
            block.append(MiniMaxH3AudioEncoderBlock(d_model, stride=stride))
        block += [
            MiniMaxH3AudioSnake1d(d_model),
            weight_norm(nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.block(hidden_states)


class MiniMaxH3AudioGeGluMlp(nn.Module):
    r"""Pre-norm GeGLU MLP used inside the attention projection block."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act(self.w0(hidden_states)) * self.w1(hidden_states)
        return self.w2(hidden_states)


class MiniMaxH3AudioCausalAttention(nn.Module):
    r"""Causal self-attention that narrows the feature width from `in_dim` to `out_dim`.

    QKV is a single bias-less `nn.Linear`; query and value biases are separate parameters and the key bias is a frozen
    zero buffer (`zero_k_bias`), exactly as stored in the checkpoint. Heads are `in_dim // num_heads` wide; instead of
    being concatenated they are **mean-pooled away**, and the remaining head dimension is adaptively average-pooled
    down to `out_dim`.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = F.linear(
            input=hidden_states,
            weight=self.qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        )
        query, key, value = (
            qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4).unbind(0)
        )
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=True)
        # The heads are mean-pooled away instead of being concatenated, and the head dimension that
        # remains is adaptively average-pooled down to `out_dim`.
        hidden_states = torch.mean(hidden_states, dim=2)
        hidden_states = F.adaptive_avg_pool1d(hidden_states, self.out_dim)
        return self.proj(hidden_states)


class MiniMaxH3AudioAttnProjection(nn.Module):
    r"""`pre_block`: residual causal-attention + GeGLU block that rewires `latent_dim` -> `latent_channels`."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MiniMaxH3AudioCausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = MiniMaxH3AudioGeGluMlp(in_features=out_dim, hidden_features=out_dim * mlp_ratio)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.proj(self.norm3(hidden_states)) + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3AudioAMPBlock(nn.Module):
    r"""BigVGAN anti-aliased multi-periodicity block (`AMPBlock1`).

    Each dilation contributes a `(dilated conv, dilation-1 conv)` pair, and every convolution is preceded by its own
    alias-free SnakeBeta activation.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2))
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2))
                for _ in dilation
            ]
        )
        self.activations = nn.ModuleList(
            [
                MiniMaxH3AudioActivation1d(activation=MiniMaxH3AudioSnakeBeta(channels))
                for _ in range(2 * len(dilation))
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = conv1(act1(hidden_states))
            residual = conv2(act2(residual))
            hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxH3AudioBigVGANDecoder(nn.Module):
    r"""BigVGAN decoder: `[batch_size, latent_dim, num_frames] -> [batch_size, 1, num_frames * 800]`."""

    def __init__(
        self,
        in_channels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))

        # Each upsampler is wrapped in a one-element `ModuleList` in the original checkpoint
        # (`ups.<i>.0`); the extra nesting is kept so the state dict stays a passthrough.
        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            nn.ConvTranspose1d(
                                upsample_initial_channel // (2**i),
                                upsample_initial_channel // (2 ** (i + 1)),
                                kernel,
                                rate,
                                padding=(kernel - rate) // 2,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(MiniMaxH3AudioAMPBlock(channels, kernel, tuple(dilation)))

        self.activation_post = MiniMaxH3AudioActivation1d(activation=MiniMaxH3AudioSnakeBeta(channels))
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv_pre(hidden_states)

        for i in range(self.num_upsamples):
            hidden_states = self.ups[i][0](hidden_states)
            residual = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j](hidden_states)
                residual = block if residual is None else residual + block
            hidden_states = residual / self.num_kernels

        hidden_states = self.activation_post(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        return torch.clamp(hidden_states, min=-1.0, max=1.0)


class AutoencoderKLMiniMaxH3Audio(nn.Module):
    r"""The audio autoencoder used by MiniMax-H3: a DAC-lineage convolutional encoder and a BigVGAN decoder,
    operating directly on mono 32 kHz waveforms.

    Args:
        encoder_dim: Channel width of the encoder's first convolution; doubles at every downsampling stage.
        encoder_rates: Encoder strides. Their product (`800`) is the hop length, i.e. 40 latents/s at 32 kHz.
        latent_dim: Width of the encoder trunk and of the decoder input, before/after the latent projections.
        latent_channels: Width of the diffusion latent, i.e. the `mean_proj` / `logs_proj` output channels.
        num_attention_heads: Number of heads in the causal-attention projection `pre_block`.
        decoder_dim: BigVGAN initial channel count; halved at every upsampling stage.
        decoder_rates: BigVGAN upsampling rates. Their product must equal `prod(encoder_rates)`.
        decoder_kernel_sizes: Transposed-convolution kernel size per upsampling stage.
        resblock_kernel_sizes: Kernel sizes of the parallel AMP residual blocks at each upsampling stage.
        resblock_dilation_sizes: Per-AMP-block dilations.
        sampling_rate: Waveform sampling rate.
        latents_mean: Per-channel latent mean used to normalize / denormalize latents.
        latents_std: Per-channel latent standard deviation used to normalize / denormalize latents.
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5),
        latent_dim: int = 2048,
        latent_channels: int = 32,
        num_attention_heads: int = 8,
        decoder_dim: int = 1024,
        decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2),
        decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        sampling_rate: int = 32000,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
    ):
        super().__init__()

        encoder_rates = tuple(int(rate) for rate in encoder_rates)
        decoder_rates = tuple(int(rate) for rate in decoder_rates)
        self.hop_length = math.prod(encoder_rates)
        if math.prod(decoder_rates) != self.hop_length:
            raise ValueError(
                f"`decoder_rates` must upsample by the encoder hop length {self.hop_length}, got "
                f"{math.prod(decoder_rates)}."
            )
        if latent_dim % latent_channels != 0:
            raise ValueError(
                f"`latent_dim` ({latent_dim}) must be a multiple of `latent_channels` ({latent_channels})."
            )

        self.sampling_rate = sampling_rate
        self.latent_channels = latent_channels
        # Not in the checkpoint, so non-persistent to keep `load_state_dict(strict=True)` a passthrough.
        self.register_buffer(
            "latents_mean",
            torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1) if latents_mean is not None else None,
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1) if latents_std is not None else None,
            persistent=False,
        )

        self.encoder = MiniMaxH3AudioEncoder(d_model=encoder_dim, strides=encoder_rates, d_latent=latent_dim)
        self.pre_block = MiniMaxH3AudioAttnProjection(latent_dim, latent_channels, num_heads=num_attention_heads)
        self.mean_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.logs_proj = nn.Conv1d(latent_channels, latent_channels, 1)

        self.dec_in_proj = nn.Conv1d(latent_channels, latent_dim, 1)
        self.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=latent_dim,
            upsample_initial_channel=decoder_dim,
            upsample_rates=decoder_rates,
            upsample_kernel_sizes=tuple(int(kernel) for kernel in decoder_kernel_sizes),
            resblock_kernel_sizes=tuple(int(kernel) for kernel in resblock_kernel_sizes),
            resblock_dilation_sizes=tuple(tuple(int(d) for d in dilation) for dilation in resblock_dilation_sizes),
        )

    @classmethod
    def from_pretrained(cls, folder: str | Path) -> "AutoencoderKLMiniMaxH3Audio":
        r"""Build from a diffusers-format folder: `config.json` plus one or more `*.safetensors` shards."""
        from safetensors.torch import load_file

        folder = Path(folder)
        with open(folder / "config.json") as f:
            config = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        model = cls(**config)

        shards = sorted(folder.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors weights found in {folder}.")
        state_dict: dict[str, Tensor] = {}
        for shard in shards:
            state_dict.update(load_file(shard))
        model.load_state_dict(state_dict, strict=True)
        return model

    def encode(self, waveform: Tensor) -> MiniMaxH3AudioDiagonalGaussianDistribution:
        r"""Encode a mono waveform `[batch_size, 1, samples]` into the latent posterior over
        `[batch_size, latent_channels, samples / 800]`.

        The waveform is right-padded to a multiple of `hop_length` (800 samples) first. MiniMax-H3 always consumes the
        posterior **mean** (`posterior.mode()`) — the `logs_proj` head is never evaluated by the reference pipeline.
        """
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"`waveform` must have shape [batch_size, 1, samples], got {tuple(waveform.shape)}.")

        right_pad = math.ceil(waveform.shape[-1] / self.hop_length) * self.hop_length - waveform.shape[-1]
        if right_pad > 0:
            waveform = F.pad(waveform, (0, right_pad))

        encoder_dtype = next(self.encoder.parameters()).dtype
        hidden_states = self.encoder(waveform.to(encoder_dtype))
        hidden_states = self.pre_block(hidden_states.transpose(1, 2)).transpose(1, 2)
        mean, logs = self.mean_proj(hidden_states), self.logs_proj(hidden_states)
        if encoder_dtype != torch.float32:
            mean, logs = mean.float(), logs.float()

        return MiniMaxH3AudioDiagonalGaussianDistribution(mean, logs)

    def decode(self, latents: Tensor) -> Tensor:
        r"""Decode denormalized latents `[batch_size, latent_channels, num_frames]` into a waveform of shape
        `[batch_size, 1, num_frames * 800]`, clamped to `[-1, 1]`."""
        if latents.ndim != 3:
            raise ValueError(
                f"`latents` must have shape [batch_size, latent_channels, num_frames], got {tuple(latents.shape)}."
            )

        decoder_dtype = next(self.decoder.parameters()).dtype
        decoded = self.decoder(self.dec_in_proj(latents.to(decoder_dtype)))
        if decoder_dtype != torch.float32:
            decoded = decoded.float()
        return decoded

    def forward(
        self,
        waveform: Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""Encode then decode a mono waveform `[batch_size, 1, samples]`. MiniMax-H3 uses the posterior mode."""
        posterior = self.encode(waveform)
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(latents)

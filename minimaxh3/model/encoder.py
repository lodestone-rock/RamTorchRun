"""encoder.py — MiniMax-H3's Qwen3-VL-32B text conditioner.

H3 conditions on `hidden_states[50]` of a Qwen3-VL-32B — the output of
decoder layer 49, *pre*-final-norm — not the final hidden state. The released
weights were trained against that tap, so the stack is truncated at layer 50
and the final norm, lm_head and vision tower are dropped (a 32B conditioner
becomes a 25B one).

The t2va presentation is the prompt verbatim: no chat template, no special
tokens. With no vision blocks, `mm_token_type_ids` is all-text and Qwen3-VL's
mrope position computation reduces to a plain arange on every axis — which is
what both `encode` below and the chunked encoder in `model/chunks.py` build,
so the two paths are numerically identical. A presentation with vision blocks
(fl2va keyframe labels) needs the full `Qwen3VLModel` path with
`mm_token_type_ids` and is left for the fl2va port.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .dit import TEXT_TAG

TEXT_ENCODER_LAYER = 50


class H3CaptionTokenizer:
    """CPU-side tokenization of a t2va presentation: the prompt verbatim, no
    chat template, no special tokens, no padding."""

    def __init__(self, folder: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(folder)

    def __call__(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(input_ids (1, L), mask (1, L) bool, token_tags (L,))`."""
        if not isinstance(prompt, str):
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so `prompt` must be a "
                f"single string, got {type(prompt)}."
            )
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        tags = torch.full((len(ids),), TEXT_TAG, dtype=torch.long)
        return input_ids, mask, tags


def load_text_encoder(folder: str, dtype: torch.dtype = torch.bfloat16):
    """Load the Qwen3-VL conditioner, truncated to the layers H3 reads.

    Returns a `Qwen3VLForConditionalGeneration` whose language model holds only
    decoder layers `0 .. layer_index - 1` with the final norm replaced by
    identity, so its `last_hidden_state` IS `hidden_states[layer_index]` of the
    full model.
    """
    from transformers import Qwen3VLForConditionalGeneration

    qwen = Qwen3VLForConditionalGeneration.from_pretrained(folder, dtype=dtype)
    qwen = qwen.eval().requires_grad_(False)
    truncate_text_encoder(qwen, TEXT_ENCODER_LAYER)
    return qwen


def truncate_text_encoder(qwen, layer_index: int = TEXT_ENCODER_LAYER) -> None:
    """Drop everything past decoder layer `layer_index - 1`, the final norm,
    the lm_head and the vision tower, in place.

    The norm becomes `nn.Identity` rather than disappearing so
    `last_hidden_state` stays the pre-norm tap H3 conditions on (HF applies
    the final norm before reporting the last hidden state).
    """
    text_model = qwen.model.language_model
    del text_model.layers[layer_index:]
    text_model.norm = nn.Identity()
    qwen.lm_head = nn.Identity()
    qwen.model.visual = nn.Identity()
    torch.cuda.empty_cache()


@torch.no_grad()
def encode(qwen, input_ids: torch.Tensor, mask: torch.Tensor, device) -> torch.Tensor:
    """Monolithic encode: `(1, L)` ids -> `(1, L, 5120)` hidden state at layer 50."""
    out = qwen.model.language_model(
        input_ids=input_ids.to(device),
        attention_mask=mask.to(device),
        use_cache=False,
    )
    return out.last_hidden_state

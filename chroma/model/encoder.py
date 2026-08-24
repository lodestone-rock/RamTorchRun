from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TextEncoderConfig:
    model_id: str
    subfolder: str = "text_encoder"
    tokenizer_subfolder: str = "tokenizer"
    max_length: int = 512


class T5Conditioner(torch.nn.Module):
    """Monolithic T5-XXL conditioner (single-GPU inference path).

    Chroma conditions on the raw T5 encoder output. Prompts are padded to a
    FIXED ``max_length`` (flow convention), so the text length is static per
    run — that is what lets the chunked trainers fix ``txtlen`` once per step.
    """

    def __init__(
        self,
        version: str = "lodestones/Chroma1-HD",
        subfolder: str = "text_encoder",
        tokenizer_subfolder: str = "tokenizer",
        max_length: int = 512,
    ):
        super().__init__()
        from transformers import T5EncoderModel, T5TokenizerFast

        self.t5 = T5EncoderModel.from_pretrained(version, subfolder=subfolder)
        self.t5 = self.t5.eval().requires_grad_(False)
        self.tokenizer = T5TokenizerFast.from_pretrained(
            version, subfolder=tokenizer_subfolder
        )
        self.max_length = max_length

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.t5.device)
            mask = inputs["attention_mask"].bool()
            out = self.t5(input_ids=inputs["input_ids"], attention_mask=mask)
            return out.last_hidden_state, mask

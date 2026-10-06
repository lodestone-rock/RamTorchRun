"""qwen_e2e.py — Qwen3-VL + K2 DiT as one trainable module ("pseudo-VAE").

Two objectives share the Qwen weights:

- caption:   Qwen-VL(image) -> scene_graph JSON, plain autoregressive loss
             with an image prefix, prompted with Qwen's native key-value
             extraction format (``KIE_PROMPT``).
- diffusion: DiT(noisy latent, Qwen(JSON text, no image), t), flow-matching
             loss whose gradient flows back through Qwen.

``encode_text`` is the trainable twin of ``Qwen3VLConditioner.forward``: same
template, same padding layout ([prefix + text | pads | suffix]), same stacked
``select_layers`` hidden states, same prefix slice — so a zero-init LoRA
reproduces the frozen conditioner bitwise and the DiT sees its familiar input.

Tokenization lives in ``E2EBatchBuilder`` and runs in the MAIN thread: HF fast
tokenizers are not safe to share across the MultiGPUWrapper worker threads.
"""
from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

# scene_graph keys with empty values, in dataset order. The template rendered
# into the caption prompt AND the key order every target is re-serialized in.
SCENE_GRAPH_SCHEMA: dict = {
    "artist": "",
    "characters": [],
    "species": [],
    "subject_count": 0,
    "subjects": [{
        "anchor": "",
        "species": "",
        "body": "",
        "hair": "",
        "eyes": "",
        "clothing": [],
        "pose": "",
        "expression": "",
        "position": "",
        "acts": [],
    }],
    "setting": "",
    "background": "",
    "lighting": "",
    "objects": [],
    "text_in_image": [],
    "watermarks": [],
    "perspective": "",
    "style": "",
    "nsfw_level": "",
}
_SUBJECT_KEYS = list(SCENE_GRAPH_SCHEMA["subjects"][0])

KIE_PROMPT = (
    "Extract the key-value information in the format:\n"
    + json.dumps(SCENE_GRAPH_SCHEMA, ensure_ascii=False)
    + ".\nOutput only the JSON."
)

# Qwen3VLConditioner's template (krea2/model/encoder.py) — kept identical.
COND_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
COND_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
COND_PREFIX_IDX = 34
COND_SUFFIX_START_IDX = 5
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


def _ordered(obj: dict, keys: list[str]) -> dict:
    out = {k: obj[k] for k in keys if k in obj}
    out.update({k: v for k, v in obj.items() if k not in out})
    return out


def serialize_scene_graph(raw: str | None) -> str:
    """Re-serialize a scene_graph JSON string: one line, schema key order
    (unknown keys appended), ``json.dumps`` default separators — the same
    layout as the template in ``KIE_PROMPT``. Unparseable input passes
    through stripped; empty input returns ""."""
    if raw is None:
        return ""
    raw = str(raw).strip()
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)
    obj = _ordered(obj, list(SCENE_GRAPH_SCHEMA))
    if isinstance(obj.get("subjects"), list):
        obj["subjects"] = [_ordered(s, _SUBJECT_KEYS) if isinstance(s, dict) else s
                           for s in obj["subjects"]]
    return json.dumps(obj, ensure_ascii=False)


def tensor_to_pil(img: Tensor) -> Image.Image:
    """[3, H, W] float in [-1, 1] -> RGB PIL."""
    arr = ((img.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round()
    return Image.fromarray(arr.to(torch.uint8).permute(1, 2, 0).numpy())


class E2EBatchBuilder:
    """CPU-side batch construction for both objectives (main thread only)."""

    def __init__(self, model_id: str, max_text_len: int = 1024,
                 caption_max_pixels: int = 512 * 512,
                 caption_min_pixels: int = 256 * 256):
        from transformers import AutoProcessor, AutoTokenizer, Qwen2TokenizerFast
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.suffix_tokenizer = Qwen2TokenizerFast.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_text_len = max_text_len
        self.size = {"longest_edge": int(caption_max_pixels),
                     "shortest_edge": int(caption_min_pixels)}
        tok = self.processor.tokenizer
        self.assistant_start = tok("<|im_start|>assistant\n")["input_ids"]
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    # -- diffusion side: Qwen3VLConditioner's exact layout ---------------------
    def text(self, jsons: list[str], pad_to_max: bool = False) -> tuple[Tensor, Tensor]:
        """Returns (input_ids, mask) for ``E2EModel.encode_text``. ``pad_to_max``
        reproduces the conditioner's fixed-length padding (parity checks);
        the default pads to the batch's longest row."""
        text = [COND_PREFIX + j for j in jsons]
        suffix = self.suffix_tokenizer(text=[COND_SUFFIX] * len(text), return_tensors="pt")
        inputs = self.tokenizer(
            text, truncation=True, return_length=False,
            return_overflowing_tokens=False,
            padding="max_length" if pad_to_max else "longest",
            max_length=self.max_text_len + COND_PREFIX_IDX - COND_SUFFIX_START_IDX,
            return_tensors="pt",
        )
        ids = torch.cat([inputs["input_ids"], suffix["input_ids"]], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(),
                          suffix["attention_mask"].bool()], dim=1)
        return ids, mask

    # -- caption side: chat template + image, labels on the answer only --------
    def _messages(self, answer: str | None) -> list[dict]:
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": KIE_PROMPT}]}]
        if answer is not None:
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": answer}]})
        return msgs

    def caption(self, images: Tensor, jsons: list[str]) -> dict[str, Tensor]:
        """images [B, 3, H, W] in [-1, 1]; jsons already serialized. Returns
        the processor's tensors plus ``labels`` (-100 outside the answer +
        its closing ``<|im_end|>``)."""
        pil = [tensor_to_pil(im) for im in images]
        text = [self.processor.apply_chat_template(self._messages(j), tokenize=False)
                for j in jsons]
        enc = self.processor(text=text, images=pil, padding=True, return_tensors="pt",
                             images_kwargs={"size": self.size})
        enc["labels"] = self.answer_labels(enc["input_ids"], enc["attention_mask"])
        return dict(enc)

    def caption_prompt(self, image: Tensor) -> dict[str, Tensor]:
        """Generation input for one image (preview)."""
        text = self.processor.apply_chat_template(
            self._messages(None), tokenize=False, add_generation_prompt=True)
        return dict(self.processor(text=[text], images=[tensor_to_pil(image)],
                                   return_tensors="pt",
                                   images_kwargs={"size": self.size}))

    def answer_labels(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        labels = torch.full_like(input_ids, -100)
        a = self.assistant_start
        n = len(a)
        for r in range(input_ids.shape[0]):
            row = input_ids[r].tolist()
            start = None
            for i in range(len(row) - n, -1, -1):
                if row[i:i + n] == a:
                    start = i + n
                    break
            if start is None:
                continue
            end = start
            while end < len(row) and attention_mask[r, end] and row[end] != self.im_end_id:
                end += 1
            if end < len(row) and row[end] == self.im_end_id:
                end += 1
            labels[r, start:end] = input_ids[r, start:end]
        return labels

    def decode(self, ids: Tensor) -> str:
        return self.processor.tokenizer.decode(ids, skip_special_tokens=True)


class E2EModel(nn.Module):
    """The wrapped module: ``dit`` (SingleStreamDiT) + ``qwen``
    (Qwen3VLForConditionalGeneration). One module so MultiGPUWrapper's ZeRO-1
    shards both LoRA sets through a single optimizer per GPU."""

    def __init__(self, dit: nn.Module, qwen: nn.Module,
                 select_layers: tuple[int, ...] = SELECT_LAYERS):
        super().__init__()
        self.dit = dit
        self.qwen = qwen
        self.select_layers = tuple(select_layers)

    def encode_text(self, input_ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Grad-enabled Qwen3VLConditioner.forward on pre-tokenized ids.
        Runs ``qwen.model`` (no lm_head: its 151k-vocab logits are unused)."""
        states = self.qwen.model(input_ids=input_ids, attention_mask=mask,
                                 output_hidden_states=True)
        hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
        return hiddens[:, COND_PREFIX_IDX:], mask[:, COND_PREFIX_IDX:]

    def caption_loss(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor,
                     pixel_values: Tensor, image_grid_thw: Tensor,
                     mm_token_type_ids: Tensor | None = None) -> Tensor:
        """Next-token CE over the answer span. lm_head runs only on the
        labelled positions (a few hundred rows, not B x L x 151k)."""
        kw = {} if mm_token_type_ids is None else {"mm_token_type_ids": mm_token_type_ids}
        out = self.qwen.model(input_ids=input_ids, attention_mask=attention_mask,
                              pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kw)
        h = out.last_hidden_state[:, :-1]
        y = labels[:, 1:]
        sel = y != -100
        logits = self.qwen.lm_head(h[sel])
        return F.cross_entropy(logits.float(), y[sel])

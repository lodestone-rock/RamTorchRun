"""check_e2e_qwen.py — invariants for the end-to-end Qwen3-VL + DiT trainer.

Tiny random Qwen3-VL (real tokenizer/processor, 36 text layers so the 12
select_layers exist) + tiny DiT, on CPU:

  1. the unchanged Qwen3VLConditioner and E2EModel.encode_text agree bitwise
     (template, padding layout, layer stack, prefix slice) — before AND after
     LoRA injection (zero-init B = exact no-op)
  2. diffusion loss: grads on the Qwen LM LoRA and the DiT LoRA, None on the
     vision LoRA (and on the unused last LM layer — the documented caveat)
  3. caption loss: grads on the vision + LM LoRA, None on the DiT
  4. labels cover exactly the answer + its closing <|im_end|>
  5. lm_head is still tied to the embeddings and carries no adapter
  6. serialize_scene_graph is lossless and puts keys in schema order
  7. schema coverage over real scene_graph rows from every source

Run: .venv/bin/python krea2/tools/check_e2e_qwen.py [--rows-per-source N]
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from einops import rearrange

import krea2.model.encoder as enc_mod
from krea2.model.lora import LoRALinear, inject_lora
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.qwen_e2e import (
    SCENE_GRAPH_SCHEMA, E2EBatchBuilder, E2EModel, serialize_scene_graph,
)
from krea2.model.sampling import prepare

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATA = "/mnt/datapool_u2/lodestone/mass_caption_v2/out/trainer_samples_v3"
FAILURES = []


def ok(name, cond, detail=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))


def tiny_qwen():
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    t = cfg.text_config
    t.hidden_size, t.intermediate_size = 64, 128
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.rope_parameters = dict(t.rope_parameters, mrope_section=[4, 2, 2])
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 4, 64, 128, 4
    v.out_hidden_size = 64
    v.deepstack_visual_indexes = [1, 2]
    torch.manual_seed(0)
    q = Qwen3VLForConditionalGeneration(cfg).float().eval()
    q.requires_grad_(False)
    return q


def tiny_dit():
    torch.manual_seed(1)
    d = SingleStreamDiT(SingleMMDiTConfig(
        features=128, tdim=32, txtdim=64, heads=4, multiplier=2, layers=2, patch=2,
        channels=16, kvheads=2, txtlayers=12, txtheads=4, txtkvheads=4))
    with torch.no_grad():
        for p in d.parameters():   # break any zero-init head so grads reach upstream
            p.copy_(torch.randn_like(p) * 0.05)
    return d


def lora_grads(module, prefix):
    out = {}
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear) and name.startswith(prefix):
            out[name] = m.lora_B.grad
    return out


def nonzero(g):
    return g is not None and float(g.abs().max()) > 0


def schema_coverage(rows_per_source: int):
    import pyarrow.parquet as pq
    top = list(SCENE_GRAPH_SCHEMA)
    sub = set(SCENE_GRAPH_SCHEMA["subjects"][0])
    srcs = sorted(glob.glob(os.path.join(DATA, "source=*")))
    ok("schema: found dataset sources", len(srcs) == 5, f"{len(srcs)} sources")
    missing, extra, type_bad, sub_missing, sub_extra = {}, {}, {}, {}, {}
    n = bad_json = 0
    for src in srcs:
        f = sorted(glob.glob(os.path.join(src, "*.parquet")))[0]
        col = pq.ParquetFile(f).read_row_group(0, columns=["scene_graph"]).column(0)
        for raw in col.to_pylist()[:rows_per_source]:
            if not raw:
                continue
            n += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            for k in top:
                if k not in obj:
                    missing[k] = missing.get(k, 0) + 1
                elif not isinstance(obj[k], type(SCENE_GRAPH_SCHEMA[k])) and not (
                        isinstance(SCENE_GRAPH_SCHEMA[k], int)
                        and isinstance(obj[k], (int, float))):
                    key = f"{k}:{type(obj[k]).__name__}"
                    type_bad[key] = type_bad.get(key, 0) + 1
            for k in obj:
                if k not in SCENE_GRAPH_SCHEMA:
                    extra[k] = extra.get(k, 0) + 1
            for s in obj.get("subjects") or []:
                if not isinstance(s, dict):
                    continue
                for k in sub - set(s):
                    sub_missing[k] = sub_missing.get(k, 0) + 1
                for k in set(s) - sub:
                    sub_extra[k] = sub_extra.get(k, 0) + 1
            rt = json.loads(serialize_scene_graph(raw))
            if rt != obj or list(rt)[:len([k for k in top if k in obj])] != [k for k in top if k in obj]:
                FAILURES.append("serialize round trip")
    print(f"  scene_graph rows scanned: {n} ({rows_per_source}/source), bad json {bad_json}")
    for label, d in (("missing top-level", missing), ("extra top-level", extra),
                     ("type mismatch", type_bad), ("missing subject", sub_missing),
                     ("extra subject", sub_extra)):
        if d:
            top5 = sorted(d.items(), key=lambda kv: -kv[1])[:8]
            print(f"    {label}: " + ", ".join(f"{k} {v/n:.1%}" for k, v in top5))
    ok("schema: valid JSON rate >= 99%", n and bad_json / n < 0.01, f"{bad_json}/{n}")
    worst = max(missing.values(), default=0) / max(n, 1)
    ok("schema: every template key present in >= 50% of rows", worst < 0.5,
       f"worst missing {worst:.1%}")
    ok("schema: serialize round trip lossless + schema-ordered",
       "serialize round trip" not in FAILURES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-source", type=int, default=1000)
    args = ap.parse_args()
    set_sdpa_ctx(False)

    print("[1] conditioner parity")
    q0 = tiny_qwen()
    orig = enc_mod.Qwen3VLForConditionalGeneration.from_pretrained
    enc_mod.Qwen3VLForConditionalGeneration.from_pretrained = staticmethod(lambda *_a, **_k: q0)
    try:
        cond = enc_mod.Qwen3VLConditioner(MODEL_ID, max_length=512)
    finally:
        enc_mod.Qwen3VLForConditionalGeneration.from_pretrained = orig
    builder = E2EBatchBuilder(MODEL_ID, max_text_len=512)
    texts = [serialize_scene_graph(json.dumps({"artist": "x", "objects": ["a", "b"]})),
             "", "a short caption"]
    with torch.no_grad():
        ref, ref_m = cond(texts)
        e2e0 = E2EModel(tiny_dit(), copy.deepcopy(q0))
        ids, m = builder.text(texts, pad_to_max=True)
        got, got_m = e2e0.encode_text(ids, m)
    ok("encode_text == conditioner (no LoRA), bitwise", torch.equal(ref, got)
       and torch.equal(ref_m, got_m), f"shape {tuple(got.shape)}")

    qwen = copy.deepcopy(q0)
    inject_lora(qwen, rank=4, include_substrings=("model.language_model.", "model.visual."))
    dit = tiny_dit()
    inject_lora(dit, rank=4, exclude_prefixes=("txtfusion.projector",))
    model = E2EModel(dit, qwen)
    model.requires_grad_(False)
    for m_ in model.modules():
        if isinstance(m_, LoRALinear):
            m_.lora_A.requires_grad_(True)
            m_.lora_B.requires_grad_(True)
    with torch.no_grad():
        got, _ = model.encode_text(ids, m)
    ok("encode_text == conditioner (zero-init LoRA), bitwise", torch.equal(ref, got))
    ids_l, m_l = builder.text(texts)
    ok("batch-longest padding is shorter than fixed padding",
       ids_l.shape[1] < ids.shape[1], f"{ids_l.shape[1]} vs {ids.shape[1]}")

    print("[5] lm_head")
    ok("lm_head not adapted", not isinstance(qwen.lm_head, LoRALinear))
    ok("lm_head tied to embed_tokens",
       qwen.lm_head.weight.data_ptr() == qwen.model.language_model.embed_tokens.weight.data_ptr())
    n_vis = len(lora_grads(qwen, "model.visual."))
    n_lm = len(lora_grads(qwen, "model.language_model."))
    ok("LoRA covers LM and vision", n_vis > 0 and n_lm > 0, f"lm {n_lm}, vision {n_vis}")

    print("[2] diffusion grad routing")
    model.train()
    torch.manual_seed(3)
    ids_d, m_d = builder.text(texts[:2])
    x0 = torch.randn(2, 16, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.rand(2)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * noise
    ctx, tm = model.encode_text(ids_d, m_d)
    x_tok, pos, mask = prepare(x_t, ctx.shape[1], 2, tm)
    v = model.dit(x_tok, ctx, t, pos, mask)
    tgt = rearrange(noise - x0, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    F.mse_loss(v, tgt).backward()
    lm = lora_grads(qwen, "model.language_model.")
    last = {k: g for k, g in lm.items() if ".layers.35." in k}
    rest = {k: g for k, g in lm.items() if ".layers.35." not in k}
    ok("diffusion: every LM LoRA below layer 35 has nonzero grad",
       all(nonzero(g) for g in rest.values()), f"{sum(map(nonzero, rest.values()))}/{len(rest)}")
    ok("diffusion: LM layer 35 LoRA gets no grad (unused by select_layers)",
       all(g is None for g in last.values()))
    ok("diffusion: vision LoRA grads are None",
       all(g is None for g in lora_grads(qwen, "model.visual.").values()))
    dg = lora_grads(dit, "")
    ok("diffusion: DiT LoRA grads nonzero", sum(map(nonzero, dg.values())) == len(dg),
       f"{sum(map(nonzero, dg.values()))}/{len(dg)}")
    model.zero_grad(set_to_none=True)

    print("[3] caption grad routing + [4] labels")
    imgs = torch.rand(2, 3, 256, 320) * 2 - 1
    jsons = [serialize_scene_graph(json.dumps({"artist": "abc", "subject_count": 1})),
             serialize_scene_graph(json.dumps({"style": "oil painting", "objects": ["x"]}))]
    cb = builder.caption(imgs, jsons)
    for r, j in enumerate(jsons):
        lab = cb["labels"][r]
        txt = builder.processor.tokenizer.decode(lab[lab != -100])
        ok(f"labels row {r} == answer + <|im_end|>", txt == j + "<|im_end|>", repr(txt[-40:]))
    ok("labels exclude padding", bool((cb["labels"][cb["attention_mask"] == 0] == -100).all()))
    loss = model.caption_loss(**cb)
    loss.backward()
    vg = lora_grads(qwen, "model.visual.")
    ok("caption: vision LoRA grads nonzero", sum(map(nonzero, vg.values())) == len(vg),
       f"{sum(map(nonzero, vg.values()))}/{len(vg)}")
    lg = lora_grads(qwen, "model.language_model.")
    ok("caption: LM LoRA grads nonzero", sum(map(nonzero, lg.values())) == len(lg),
       f"{sum(map(nonzero, lg.values()))}/{len(lg)}")
    ok("caption: DiT LoRA grads are None",
       all(g is None for g in lora_grads(dit, "").values()))
    ok("caption loss ~ ln(vocab) at random init", abs(float(loss) - 11.93) < 1.0,
       f"{float(loss):.3f}")

    print("[6/7] scene_graph schema")
    schema_coverage(args.rows_per_source)

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()

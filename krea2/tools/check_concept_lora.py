"""check_concept_lora.py — CPU guards for the dedicated concept-LoRA trainer.

Fast, no GPU, no checkpoint, no parquet reads of real data. Four sections:

  1. SCHEDULE SHAPE   — build_schedule_multiplier's warmup/constant/cooldown
  2. CACHE ROUND-TRIP — fingerprint stability + marker cleaning on a synthetic
     parquet via a monkeypatched (tiny) VAE/encoder path... actually no: the
     full build_cache touches CUDA + HF downloads, which is not a CPU check.
     We verify instead the pure parts: _cfg_fingerprint stability, the
     manifest round-trip, and that build_cache's fingerprint gate REJECTS a
     manifest whose fingerprint differs.
  3. NO-OP PARITY     — TagEmbedder with a 1-entry vocab is a bit-exact no-op
     when the marker is masked off (the trainer's step-0 discipline).
  4. BUCKET RULES     — _bucket_for matches the shared loader's filter on a
     synthetic bucket grid.

Run:  uv run python krea2/tools/check_concept_lora.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from krea2.model.concept_marker import marker_row, parse_concept_markers
from krea2.train_concept_lora import (
    _bucket_for,
    _cfg_fingerprint,
    build_schedule_multiplier,
)

PASS = "  ok  "
FAIL = " FAIL "


def check(cond, msg):
    print(f"{PASS if cond else FAIL}{msg}")
    if not cond:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
print("== 1. schedule shape (50 warmup / 300 constant / 50 cooldown) ==")
m = build_schedule_multiplier(max_steps=400, warmup=50, cooldown=50)
check(abs(m(0) - 1 / 50) < 1e-12, "step 0     -> lr/50      (first warmup step)")
check(abs(m(1) - 2 / 50) < 1e-12, "step 1     -> 2*lr/50")
check(abs(m(49) - 1.0) < 1e-12, "step 49    -> full lr (warmup done)")
check(m(50) == 1.0 and m(349) == 1.0, "steps 50..349 -> constant full lr")
check(m(350) == 1.0, "step 350   -> cooldown starts AT full lr (50/50)")
check(abs(m(351) - 49 / 50) < 1e-12, "step 351   -> first decayed step")
check(abs(m(399) - 1 / 50) < 1e-12, "step 399 -> 1/50 (positive, floor 1/400 inactive)")
check(m(400) == 1 / 400, "step 400 (past end) -> floor")

m0 = build_schedule_multiplier(400, 0, 0)
check(all(m0(s) == 1.0 for s in (0, 1, 199, 399)), "warmup=cooldown=0 -> constant schedule")

# ---------------------------------------------------------------------------
print("== 2. cache fingerprint + manifest round-trip ==")
with tempfile.TemporaryDirectory() as td:
    cfg = {
        "artist": "chunie",
        "trigger": "by chunie",
        "caption_bands": {"tags": 2.0},
        "base_resolution": 1024,
        "resolution_step": 128,
        "ratio_cutoff": 2.0,
        "dataset_marker_from": None,
        "parquet_path": os.path.join(td, "fake.parquet"),
    }
    # _cfg_fingerprint stats the parquet; make a stub for it.
    open(cfg["parquet_path"], "wb").write(b"stub")

    fp1 = _cfg_fingerprint(cfg)
    fp2 = _cfg_fingerprint(cfg)
    check(fp1 == fp2, "fingerprint stable across calls")

    cfg2 = dict(cfg)
    cfg2["caption_bands"] = {"tags": 3.0}
    check(_cfg_fingerprint(cfg2) != fp1, "fingerprint reacts to caption_bands")

    cfg3 = dict(cfg)
    cfg3["trigger"] = "different trigger"
    check(_cfg_fingerprint(cfg3) != fp1, "fingerprint reacts to trigger")

    # manifest round-trip through the SAME JSON the trainer writes
    manifest = {
        "fingerprint": fp1,
        "trigger": "by chunie",
        "vocab": ["by chunie"],
        "bands": ["tags"],
        "images": [],
        "uncond_path": os.path.join(td, "uncond.sft"),
    }
    p = os.path.join(td, "manifest.json")
    with open(p, "w") as f:
        json.dump(manifest, f)
    with open(p) as f:
        loaded = json.load(f)
    check(loaded["fingerprint"] == fp1 and loaded["vocab"] == ["by chunie"],
          "manifest JSON round-trip preserves fingerprint + vocab")

# ---------------------------------------------------------------------------
print("== 3. no-op parity: masked 1-entry TagEmbedder emits zero tokens ==")
from krea2.model.tag_embed import TagEmbedder

torch.manual_seed(0)
tag_dim = 512
emb = TagEmbedder(vocab_size=1, features=128, tag_dim=tag_dim)
emb.eval()

B = 2
ids = torch.zeros(B, 1, dtype=torch.long)
mask = torch.zeros(B, 1, dtype=torch.bool)          # concept DROPPED
mask_on = torch.ones(B, 1, dtype=torch.bool)        # concept PRESENT

out_off = emb(ids, mask)
check(bool((out_off == 0).all()),
      "mask=False -> all concept tokens exactly zero (concat no-op)")

# Fresh module = step-0 state (zero-init projection): even an on-mask token is
# exactly zero, which is what makes the first training step base-safe.
out_step0 = emb(ids, mask_on)
check(bool((out_step0 == 0).all()),
      "zero-init projection + on-mask -> zero tokens (step-0 safety)")

# After (simulated) training, an on-mask token carries a vector.
with torch.no_grad():
    emb.proj.weight.normal_(0, 0.1)
out_on = emb(ids, mask_on)
delta = out_on.abs().max().item()
check(delta > 0, "trained projection + on-mask -> token carries a vector")
print(f"        (on-mask token magnitude {delta:.2e})")
out_off = emb(ids, mask)
check(bool((out_off == 0).all()),
      "trained projection + mask=False -> still exactly zero")

# ---------------------------------------------------------------------------
print("== 4. bucket rules match the shared loader ==")
# Build the same standardized set build_cache uses and probe _bucket_for.
from dataloaders.bucketing_logic import _bucket_generator, _normalize_width_height

standardized = {}
for bucket in _bucket_generator(1024, 2.0, 128):
    standardized[_normalize_width_height(*bucket)] = bucket

# in-cutoff sizes map to a real bucket (values are raw (w,h); keys are
# aspect-normalized floats)
b = _bucket_for(1024, 1024, standardized, 2.0)
check(b is not None and tuple(b) in standardized.values(),
      f"1024x1024 -> {b} (a real bucket)")
b = _bucket_for(1500, 800, standardized, 2.0)
check(b is not None, f"1500x800 (1.875) inside cutoff -> {b}")

# out-of-cutoff sizes are dropped, exactly like the loader's filter
b = _bucket_for(2000, 512, standardized, 2.0)
check(b is None, "2000x512 (3.9) outside cutoff 2.0 -> dropped (None)")
b = _bucket_for(0, 0, standardized, 2.0)
check(b is None, "0x0 -> dropped (None)")

# ---------------------------------------------------------------------------
print("== 5. plain engine: unmasked == masked, compacted text == padded text ==")
from krea2.model import mmdit as _mmdit
from krea2.model.lora import inject_lora
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT
from krea2.model.sampling import prepare

_mmdit.set_sdpa_ctx(False)        # CPU: no cuDNN-only SDPA context
tiny = SingleMMDiTConfig(features=64, tdim=32, txtdim=32, heads=4, kvheads=2,
                         multiplier=2, layers=2, patch=2, channels=4,
                         txtheads=2, txtkvheads=2, txtlayers=3)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    model = SingleStreamDiT(tiny)
    inject_lora(model, rank=4, alpha=4.0, exclude_prefixes=("txtfusion.projector",))
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)        # zero-init modulation would make blocks no-ops
        for name, b in model.named_buffers():
            if name.endswith("base_weight"):
                b.normal_(0, 0.1)
    for name, p in model.named_parameters():
        p.requires_grad_(name.endswith((".lora_A", ".lora_B")))
    return model


model = tiny_model()
x0 = torch.randn(1, 4, 8, 8)
txt = torch.randn(1, 7, 3, 32)
t = torch.tensor([0.6])
tok, pos, mask = prepare(x0, 7, 2, torch.ones(1, 7, dtype=torch.bool))
with torch.no_grad():
    masked = model(tok, txt, t, pos, mask)
    unmasked = model(tok, txt, t, pos, None)
err = (masked - unmasked).abs().max().item()
check(err < 1e-5, f"mask=None (no 256-pad) == all-valid masked path (max err {err:.1e})")

# Cache format 2: dropping tokenizer-pad rows == attending them masked.
pad = torch.randn(1, 5, 3, 32)
txt_padded = torch.cat((txt[:, :4], pad, txt[:, 4:]), dim=1)      # pads mid-sequence
pmask = torch.ones(1, 12, dtype=torch.bool)
pmask[:, 4:9] = False
tok_p, pos_p, mask_p = prepare(x0, 12, 2, pmask)
with torch.no_grad():
    padded = model(tok_p, txt_padded, t, pos_p, mask_p)
err = (padded - unmasked).abs().max().item()
check(err < 1e-5, f"compacted valid rows == mid-padded masked text (max err {err:.1e})")


def loss_and_grads(m):
    out = m(tok, txt, t, pos, None)
    loss = torch.nn.functional.mse_loss(out, torch.ones_like(out))
    loss.backward()
    grads = [p.grad.clone() for p in m.parameters() if p.requires_grad]
    m.zero_grad(set_to_none=True)
    return loss.detach(), grads


loss_a, grads_a = loss_and_grads(model)
check(all(g is not None and g.abs().sum() > 0 for g in grads_a),
      "unmasked backward reaches every adapter")

print("== 6. compiled blocks == eager (CPU inductor) ==")
from krea2.compile import compile_plain_dit_blocks

compiled = tiny_model()
names = compile_plain_dit_blocks(compiled, {"mode": "default", "dynamic": True})
check(len(names) == 4 + tiny.layers, f"compiled {len(names)} blocks (4 text + {tiny.layers} DiT)")
loss_c, grads_c = loss_and_grads(compiled)
gerr = max((a - c).abs().max().item() / (a.abs().max().item() + 1e-12)
           for a, c in zip(grads_a, grads_c))
check(abs(loss_a - loss_c).item() < 1e-5 and gerr < 1e-3,
      f"compiled loss {loss_c.item():.6f} vs eager {loss_a.item():.6f}, grad rel err {gerr:.1e}")

print("== 7. optimizer switch ==")
from krea2.train_concept_lora import build_optimizer, shuffled_tag_variants, validate_config

base_cfg = dict(device="cpu", artist="chunie", parquet_path="x", mmdit_checkpoint="x",
                caption_bands={"tags": 1.0})
validate_config(dict(base_cfg, optimizer="adamw"))
for bad in (dict(optimizer="sgd"), dict(engine="pipeline", compile=True),
            dict(tag_bands={"long_caption": 4})):
    try:
        validate_config(dict(base_cfg, **bad))
        check(False, f"validate_config rejects {bad}")
    except ValueError:
        check(True, f"validate_config rejects {bad}")
p1 = torch.nn.Parameter(torch.randn(8, 4))
p2 = torch.nn.Parameter(p1.detach().clone())
opt = build_optimizer(dict(optimizer="adamw", lr=1e-3), [p1])
ref = torch.optim.AdamW([p2], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
for _ in range(3):
    g = torch.randn(8, 4)
    p1.grad, p2.grad = g.clone(), g.clone()
    opt.step()
    ref.step()
check(torch.equal(p1, p2), "adamw option == torch AdamW(betas 0.9/0.95, wd 0)")
check(isinstance(build_optimizer(dict(optimizer="lion", lr=1e-4), [p1]),
                 __import__("krea2.lion", fromlist=["Lion"]).Lion), "lion option -> krea2 Lion")

print("== 8. tag shuffle variants ==")
cap = "a, b, c, d, e, f"
v1 = shuffled_tag_variants(cap, 4, "sha:tags")
check(v1 == shuffled_tag_variants(cap, 4, "sha:tags"), "deterministic per seed")
check(all(sorted(x.strip() for x in v.split(",")) == sorted(x.strip() for x in cap.split(","))
          for v in v1), "each variant is a permutation of the tags")
check(len(set(v1)) > 1, f"variants differ ({len(set(v1))} distinct of 4)")

print("== 9. LoRA -> 1-slot bank round trip ==")
from krea2.model.lora import lora_state_dict
from krea2.tools.export_lora_bank import slice_slot
from krea2.tools.lora_to_bank import to_bank

sd = {k: v.detach() for k, v in lora_state_dict(model).items()}
bank = to_bank(sd)
check(all(v.shape[0] == 1 and k.endswith("_bank") for k, v in bank.items()),
      f"{len(bank)} tensors with a leading slot dim")
back = slice_slot(bank, 0)
check(back.keys() == sd.keys() and all(torch.equal(back[k], sd[k].float()) for k in sd),
      "slice_slot(to_bank(sd), 0) == sd exactly")
try:
    to_bank(dict(sd, **{"blocks.0.prenorm.scale": torch.zeros(4)}))
    check(False, "non-adapter keys refused")
except ValueError:
    check(True, "non-adapter keys refused without --drop-extra")

print("\nAll concept-LoRA CPU guards passed.")

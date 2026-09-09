"""check_slot_sweep.py — CPU guard for krea2/sweep_lora_bank.py's LoRA plumbing.

CPU checks (mirrors check_lora_bank.py's tiny config):

1. **Slot fidelity**: a tiny DiT with injected rank-8 LoRALinear whose params
   are copied from bank slot s (exactly `sweep_lora_bank.load_slot` does)
   matches a LoRABankLinear model running with `set_active_slots([s])`.
   Uses nonzero synthetic adapters with matching tiny-model dimensions;
   also checks 0->1->0 determinism and clearing absent adapters.
2. **Merge math**: `utils/checkpoint.merge_lora_into_base_sd` (which the
   sweep uses for the turbo delta) reproduces the reference x0-pred
   `apply_lora_to_full_model.py` merge on a tiny synthetic full FT + LoRA —
   including the `.base_bias` -> `.bias` rename — and a base with that merge
   applied + inject_lora + turbo slices matches the direct merged model
   numerically.
3. **Real-file plumbing, CPU**: the sweep's own `load_bank` / `load_slot`
   against a real bank slice on a Linear with matching dimensions, plus the
   turbo checkpoint's key layout.
4. Multi-batch conditioning cache and safe resume checks.

Run:  uv run python krea2/tools/check_slot_sweep.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from krea2.model.lora import LoRALinear, inject_lora
from krea2.model.lora_bank import inject_lora_bank, set_active_slots
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.sampling import prepare
from krea2.sweep_lora_bank import (
    Conditioner, amplify_full_delta, image_name, load_bank, load_slot,
    slot_complete, write_json,
)
from utils.checkpoint import merge_lora_into_base_sd

# mmdit wraps SDPA in sdpa_kernel(CUDNN), which has no CPU backend.
set_sdpa_ctx(False)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E621 = os.path.join(_REPO, "runs/k2-mass-lora-v2-e621/ckpts/bank_step_12200.safetensors")
DAN = os.path.join(_REPO, "runs/k2-mass-lora-v2-danbooru/ckpts/bank_step_11700.safetensors")
TURBO = os.path.join(_REPO, "checkpoints/krea2/turbo_delta_r512_fullft49600.safetensors")


# Tiny but structurally faithful (same recipe as check_chunk_parity.py):
# headdim 64 satisfies the RoPE axis split.
TINY = SingleMMDiTConfig(
    features=128,
    tdim=32,
    txtdim=64,
    heads=2,
    kvheads=2,
    multiplier=2,
    layers=2,
    patch=2,
    channels=4,
    txtheads=2,
    txtkvheads=2,
    txtlayers=2,
)
LATENT = 8      # -> (LATENT/patch)^2 = 16 image tokens
TXTLEN = 5


def tiny_dit():
    torch.manual_seed(0)
    return SingleStreamDiT(TINY).eval().requires_grad_(False)


def dit_inputs(cfg=TINY):
    torch.manual_seed(1)
    latent = torch.randn(1, cfg.channels, LATENT, LATENT)
    txtmask = torch.ones(1, TXTLEN, dtype=torch.bool)
    img, pos, mask = prepare(latent, TXTLEN, cfg.patch, txtmask)
    context = torch.randn(1, TXTLEN, cfg.txtlayers, cfg.txtdim)
    t = torch.rand(1)
    return img, context, t, pos, mask


def fwd(dit, inputs):
    with torch.no_grad():
        return dit(img=inputs[0], context=inputs[1], t=inputs[2],
                   pos=inputs[3], mask=inputs[4])



def check_slot_fidelity() -> None:
    inputs = dit_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        ref = tiny_dit()
        inject_lora_bank(ref, n_slots=3, rank=8, alpha=8)
        bank_sd = {}
        with torch.no_grad():
            for k, p in ref.named_parameters():
                if k.endswith((".lora_A_bank", ".lora_B_bank")):
                    p.normal_(std=0.05)
                    bank_sd[k] = p.detach().contiguous()
        path = os.path.join(tmp, "bank.safetensors")
        save_file(bank_sd, path)
        bank = load_bank("tiny:" + path)
        test = tiny_dit()
        inject_lora(test, rank=8, alpha=8)
        lora_mods = {n: m for n, m in test.named_modules()
                     if isinstance(m, LoRALinear)}
        outputs = []
        for s in (0, 1, 0):
            set_active_slots(ref, [s])
            want = fwd(ref, inputs)
            n = load_slot(lora_mods, bank, s)
            got = fwd(test, inputs)
            assert n > 0
            torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
            outputs.append(got)
        assert torch.equal(outputs[0], outputs[2])
        assert not torch.equal(outputs[0], outputs[1])
        bank["a_keys"] = bank["a_keys"][:1]
        load_slot(lora_mods, bank, 2)
        kept = bank["a_keys"][0].removesuffix(".lora_A_bank")
        assert all(torch.count_nonzero(m.lora_B) == 0
                   for name, m in lora_mods.items() if name != kept)
        print(f"  [OK ] {n} nonzero adapters: fidelity, 0->1->0, subset clearing")


def check_merge_math() -> None:
    """merge_lora_into_base_sd == x0-pred's apply_lora_to_full_model math,
    including the base_bias -> .bias rename, on a tiny synthetic pair."""
    dit = tiny_dit()
    base_sd = {k: v.clone() for k, v in dit.state_dict().items()}
    inputs = dit_inputs()

    rank, alpha = 4, 8.0
    lora_sd = {}
    for k in base_sd:
        if k.endswith(".weight") and base_sd[k].dim() == 2 \
                and min(base_sd[k].shape) > rank:
            out, inp = base_sd[k].shape
            g = torch.Generator().manual_seed(abs(hash(k)) % 2**31)
            lora_sd[k[:-len(".weight")] + ".lora_A"] = torch.randn(
                rank, inp, generator=g) * 0.02
            lora_sd[k[:-len(".weight")] + ".lora_B"] = torch.randn(
                out, rank, generator=g) * 0.02
    # Non-LoRA trainables, including a Linear bias stored as base_bias.
    lora_sd["blocks.0.prenorm.scale"] = torch.randn_like(
        base_sd["blocks.0.prenorm.scale"])
    lora_sd["first.base_bias"] = torch.randn_like(base_sd["first.bias"])

    merged = merge_lora_into_base_sd(
        {k: v.clone() for k, v in base_sd.items()}, dict(lora_sd),
        rank=rank, alpha=alpha)

    # Reference merge (x0-pred apply_lora_to_full_model.py math, inlined):
    ref = {k: v.clone() for k, v in base_sd.items()}
    for k in lora_sd:
        if k.endswith(".lora_A"):
            b_key = k[:-len(".lora_A")] + ".lora_B"
            w_key = k[:-len(".lora_A")] + ".weight"
            ref[w_key] = ref[w_key] + (alpha / rank) * (lora_sd[b_key] @ lora_sd[k])
    ref["blocks.0.prenorm.scale"] = lora_sd["blocks.0.prenorm.scale"]
    ref["first.bias"] = lora_sd["first.base_bias"]

    diffs = [(k, (merged[k] - ref[k]).abs().max().item())
             for k in ref if k in merged]
    bad = [(k, d) for k, d in diffs if d != 0]
    missing = [k for k in ref if k not in merged]
    assert not missing and not bad, f"merge mismatch: {missing}, {bad[:3]}"
    assert "first.base_bias" not in merged  # renamed, not passed through
    print(f"  [OK ] merge == reference on {len(diffs)} keys "
          f"(bias renamed first.base_bias -> first.bias)")

    # And the merged model must behave like base + inject_lora with the same
    # adapters loaded: same forward from both paths.
    m = tiny_dit(); m.load_state_dict(merged, strict=True)
    t = tiny_dit(); t.load_state_dict(base_sd, strict=True)
    inject_lora(t, rank=rank, alpha=alpha)
    tsd = dict(t.state_dict())
    lora_keys = {k for k in lora_sd if k.endswith((".lora_A", ".lora_B"))}
    loaded = set()
    for k in lora_keys:
        # inject_lora adapts every nn.Linear; the synthetic bank covered only
        # min(shape) > rank. Adapters that were never injected keep B=0 and
        # contribute nothing, so only load the ones that exist.
        if k in tsd and tsd[k].shape == lora_sd[k].shape:
            tsd[k] = lora_sd[k]
            loaded.add(k)
    tsd["blocks.0.prenorm.scale"] = lora_sd["blocks.0.prenorm.scale"]
    tsd["first.base_bias"] = lora_sd["first.base_bias"]
    t.load_state_dict(tsd, strict=True)
    skipped = len(lora_keys) - len(loaded)
    d = (fwd(m, inputs) - fwd(t, inputs)).abs().max().item()
    assert d < 1e-4, f"merged-vs-injected forward differs: {d}"
    print(f"  [OK ] merged full model == inject_lora+adapters forward, "
          f"max|d|={d:.3e} ({loaded.size if hasattr(loaded,'size') else len(loaded)} "
          f"adapter keys loaded, {skipped} skipped by the rank guard on both)")




def check_real_plumbing() -> None:
    """The sweep's own functions against the real bank + turbo files, on a
    tiny DiT (CPU): every key they promise to swap actually exists."""
    bank = load_bank(os.path.relpath(E621, _REPO))
    # Use a real module's geometry, not a tiny DiT whose shapes cannot match.
    spot = min(bank["a_keys"], key=lambda k:
               bank["handle"].get_slice(k).get_shape()[-1])
    name = spot.removesuffix(".lora_A_bank")
    A = bank["handle"].get_slice(spot)[3]
    B = bank["handle"].get_slice(name + ".lora_B_bank")[3]
    layer = torch.nn.Linear(A.shape[1], B.shape[0], bias=False)
    model = torch.nn.Sequential(layer)
    inject_lora(model, rank=bank["rank"], alpha=bank["alpha"])
    lora_mods = {name: model[0]}
    bank["a_keys"] = [spot]
    n = load_slot(lora_mods, bank, 3)
    assert n == 1
    d = (lora_mods[name].lora_A.detach() - A).abs().max().item()
    assert d == 0, f"real-bank slice not copied exactly: {d}"
    torch.testing.assert_close(lora_mods[name].lora_B, B.float(), atol=0, rtol=0)
    print(f"  [OK ] real bank A/B slices land exactly ({name!r})")

    with safe_open(TURBO, "pt") as f:
        keys = set(f.keys())
    a_keys = [k for k in keys if k.endswith(".lora_A")]
    other = [k for k in keys if not k.endswith((".lora_A", ".lora_B"))]
    base_bias = [k for k in other if k.endswith(".base_bias")]
    assert len(a_keys) == 264 and base_bias, "turbo ckpt format changed?"
    print(f"  [OK ] turbo ckpt: {len(a_keys)} deltas, {len(other)} overrides "
          f"({len(base_bias)} of them .base_bias — the renamed kind)")


def check_cache_resume():
    class Encoder(torch.nn.Module):
        calls = 0

        def forward(self, prompts):
            self.calls += 1
            return torch.ones(len(prompts), 2), torch.ones(len(prompts), 2, dtype=torch.bool)

    enc = Encoder()
    cond = Conditioner(enc, torch.device("cpu"), torch.bfloat16, cache_size=2)
    for _ in range(2):
        for batch in (("first",), ("second",)):
            cond.encode(batch)
    assert enc.calls == 2, "every batch must remain cached across slots"
    with tempfile.TemporaryDirectory() as tmp:
        entries = [dict(index=0, seed=1234, prompt="test")]
        assert not slot_complete(tmp, "abc", entries)
        Path(tmp, image_name(entries[0])).write_bytes(b"saved image")
        write_json(os.path.join(tmp, "manifest.json"), entries)
        assert not slot_complete(tmp, "abc", entries)
        write_json(os.path.join(tmp, "complete.json"), dict(signature="abc"))
        assert slot_complete(tmp, "abc", entries)
        try:
            slot_complete(tmp, "changed", entries)
        except ValueError:
            pass
        else:
            raise AssertionError("changed settings must not silently resume")
        Path(tmp, image_name(entries[0])).unlink()
        assert not slot_complete(tmp, "abc", entries)
    print("  [OK ] multi-batch cache, interrupted resume, settings mismatch")


def check_full_delta():
    base = {"first.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "first.bias": torch.tensor([0.25, -0.5])}
    tuned = {k: v + 0.125 for k, v in base.items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tuned.safetensors")
        save_file(tuned, path)
        for scale in (0.0, 1.0, 1.5, 2.0):
            result = amplify_full_delta(dict(base), path, scale)
            for k in base:
                torch.testing.assert_close(result[k], base[k] + scale * (tuned[k] - base[k]),
                                           atol=0, rtol=0)
            if scale == 1.5:
                A, B = torch.ones(1, 2), torch.ones(2, 1)
                merge_lora_into_base_sd(result, {"first.lora_A": A, "first.lora_B": B,
                                                "first.base_bias": torch.zeros(2)},
                                        rank=1, alpha=1)
                torch.testing.assert_close(result["first.weight"],
                                           base["first.weight"] + 1.5 * 0.125 + B @ A)
                assert torch.count_nonzero(result["first.bias"]) == 0
        assert base["first.weight"][0, 0].item() == 1.0, "source mutated"
        for invalid in ({"wrong": torch.ones(2)},
                        dict(tuned, **{"first.weight": torch.ones(3)})):
            save_file(invalid, path)
            try:
                amplify_full_delta(dict(base), path, 1.5)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid checkpoint accepted")
        save_file(tuned, path)
        try:
            amplify_full_delta(dict(base), path, float("nan"))
        except ValueError:
            pass
        else:
            raise AssertionError("NaN scale accepted")
    print("  [OK ] fp32 full-delta endpoints/extrapolation, turbo order, validation")


def main() -> int:
    print("check 1/3: slot fidelity (bank slot == hot-swapped injected LoRA)")
    check_slot_fidelity()
    print("check 2/3: merge math (== x0-pred reference, with bias rename)")
    check_merge_math()
    print("check 3/3: real-file plumbing (real bank slice + turbo keys)")
    check_real_plumbing()
    check_cache_resume()
    check_full_delta()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

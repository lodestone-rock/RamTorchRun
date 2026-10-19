"""CPU check of train_sft's ``parallelism: "zero"`` (utils/zero_sharding.py) against one process.

Tiny MOSS model, hybrid Muon-AdamW, grad accumulation, global clip (made active), q/k decay-to-one, EMA.
gloo ranks each see their own micro-batches; the reference sees all of them in one process. Checks:

  - world 1 and 3: weights, the union of the owned EMA shards and the union of the owned optimizer states
    match the reference after several updates (and every parameter has exactly one owner)
  - ``swapped``: inside it every replica holds the full EMA; on exit the live weights are back
  - zero.clip "local": the returned norm is still the global one; each owner's grads end <= max_grad_norm
  - comm_dtype bf16 + muon state_dtype bf16: close to the reference
  - DCP with per-rank optimizer / EMA keys: resume at the same world is bit-exact with uninterrupted
  - --init-from semantics: a world-3 checkpoint loads into world 2 (different owners): weights and every
    owned optimizer state equal the reference's

uv run python moss/tools/check_zero_sync.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

import utils.muon
from moss.model.transformer import MOSSConfig, MOSSImageTransformer
from moss.train_sft import EMA, ZERO_STATE_OPTIONS as FLAT, build_optimizers, muon_eligible
from utils.zero_sharding import ZeroSync


def orthogonalize_fp32(g, steps=5):
    """utils.muon.orthogonalize in fp32. The bf16 original amplifies an fp32 ulp of difference (the grad
    sum's reduction order across ranks) up to ~3.4^5 along small singular directions, ~10% of the update,
    which would drown the sharding errors this check looks for."""
    x = g.float()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = utils.muon.NS_COEFFICIENTS
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    return x.mT if tall else x


utils.muon.orthogonalize = orthogonalize_fp32

CFG = {"optimizer": "muon", "parallelism": "zero", "lr": 3e-3, "betas": [0.9, 0.95], "weight_decay": 0.01,
       "max_grad_norm": 0.05, "qk_norm_decay_to_one": 0.5}
ACCUM, UPDATES, DECAY = 2, 3, 0.8


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = MOSSConfig(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=40, freq_dim=32, latent_channels=8)
    return MOSSImageTransformer(config).float().requires_grad_(True)


def micro_loss(model, update, rank, micro):
    g = torch.Generator().manual_seed(10_000 * update + 100 * rank + micro)
    x = torch.randn(2, 8, 4, 4, generator=g)
    t = torch.rand(2, generator=g) * 1000
    ctx = [torch.randn(5, 40, generator=g), torch.randn(3, 40, generator=g)]
    return F.mse_loss(model(x, t, ctx), torch.randn(2, 8, 4, 4, generator=g))


def optimizers_for(model, cfg, owned=None):
    optimizers = build_optimizers(model, cfg, owned)
    for o in optimizers:
        if isinstance(o, torch.optim.AdamW):
            for group in o.param_groups:
                group["fused"] = None
    return optimizers


def finish(model, optimizers, ema, cfg):
    for o in optimizers:
        o.step()
    with torch.no_grad():
        for p in (model.layers[-1].q_norm.weight, model.layers[-1].k_norm.weight):
            p.sub_(p - 1, alpha=cfg["lr"] * cfg["qk_norm_decay_to_one"])
    ema.update(model)


def states(model, optimizers):
    names = {id(p): n for n, p in model.named_parameters()}
    return {names[id(p)]: {k: v.detach().float().clone() for k, v in o.state[p].items() if torch.is_tensor(v)
                           and v.ndim > 0}
            for o in optimizers for p in o.state}


def reference(world, updates, cfg):
    model = tiny_model()
    optimizers = optimizers_for(model, {**cfg, "parallelism": "ddp"})
    ema = EMA(model, DECAY)
    norms = []
    for update in range(updates):
        for o in optimizers:
            o.zero_grad(set_to_none=True)
        for rank in range(world):
            for micro in range(ACCUM):
                (micro_loss(model, update, rank, micro) / ACCUM / world).backward()
        norms.append(torch.nn.utils.clip_grad_norm_(list(model.parameters()), cfg["max_grad_norm"]).item())
        finish(model, optimizers, ema, cfg)
    return {"params": {n: p.detach().clone() for n, p in model.named_parameters()},
            "ema": {n: s.clone() for n, s in ema.shadow.items()}, "states": states(model, optimizers),
            "norms": norms}


def setup(rank, world, cfg, seed=0):
    model = tiny_model(seed)
    named = list(model.named_parameters())
    muon = [i for i, (n, p) in enumerate(named) if muon_eligible(n, p)]
    zero = ZeroSync(named, rank, world, groups=[muon, sorted(set(range(len(named))) - set(muon))], mode="loop",
                    comm_dtype=cfg.get("comm_dtype"), bucket_mb=1)
    owned = set(zero.owned_names())
    return model, zero, optimizers_for(model, cfg, owned), EMA(model, DECAY, owned)


def run_updates(model, zero, optimizers, ema, cfg, rank, updates, clip="global"):
    norms, local_norms = [], []
    for update in updates:
        for o in optimizers:
            o.zero_grad(set_to_none=True)
        for micro in range(ACCUM):
            (micro_loss(model, update, rank, micro) / ACCUM).backward()
        zero.reduce_grads()
        assert all((p.grad is None) != mine for p, mine in zip(zero.params, zero.owned_mask)), "grad ownership"
        norms.append(zero.clip(cfg["max_grad_norm"], clip).item())
        owned_grads = [p.grad for p in zero.owned()]
        local_norms.append(torch.stack([g.norm() for g in owned_grads]).norm().item() if owned_grads else 0.0)
        finish(model, optimizers, ema, cfg)
        zero.broadcast_params()
    return norms, local_norms


def worker(rank, world, port, folder, task, cfg):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    model, zero, optimizers, ema = setup(rank, world, cfg)
    out = {"owners": zero.owners, "owned": zero.owned_names()}
    if task in ("parity", "local"):
        out["norms"], out["local_norms"] = run_updates(model, zero, optimizers, ema, cfg, rank, range(UPDATES),
                                                       clip="global" if task == "parity" else "local")
        out["params"] = {n: p.detach().clone() for n, p in model.named_parameters()}
        out["ema"] = {n: s.clone() for n, s in ema.shadow.items()}
        out["states"] = states(model, optimizers)
        with zero.swapped(ema, model):
            out["swapped"] = {n: p.detach().clone() for n, p in model.named_parameters()}
        out["restored"] = all(torch.equal(p.detach(), out["params"][n]) for n, p in model.named_parameters())
    elif task == "resume":
        run_updates(model, zero, optimizers, ema, cfg, rank, range(2))
        model_state, optim_state = get_state_dict(model, optimizers, options=FLAT)
        dcp.save({"model": model_state, "optim": optim_state, "ema": ema.state_dict()},
                 checkpoint_id=os.path.join(folder, "ckpt"))
        run_updates(model, zero, optimizers, ema, cfg, rank, [2])
        uninterrupted = {n: p.detach().clone() for n, p in model.named_parameters()}
        model, zero, optimizers, ema = setup(rank, world, cfg, seed=5)
        load_into(model, optimizers, ema, os.path.join(folder, "ckpt"))
        assert ema.count == 2
        run_updates(model, zero, optimizers, ema, cfg, rank, [2])
        out["resume_err"] = max((p.detach() - uninterrupted[n]).abs().max().item()
                                for n, p in model.named_parameters())
    elif task == "init_from":
        load_into(model, optimizers, ema, os.path.join(folder, "ckpt"))
        out["params"] = {n: p.detach().clone() for n, p in model.named_parameters()}
        out["states"] = states(model, optimizers)
    torch.save(out, os.path.join(folder, f"{task}-w{world}-r{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


def load_into(model, optimizers, ema, path):
    model_state, optim_state = get_state_dict(model, optimizers, options=FLAT)
    state = {"model": model_state, "optim": optim_state, "ema": ema.state_dict()}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(model, optimizers, model_state_dict=state["model"], optim_state_dict=state["optim"],
                   options=FLAT)
    ema.load_state_dict(state["ema"])


PORT = [29720]


def launch(world, folder, task, cfg):
    PORT[0] += 1
    mp.spawn(worker, args=(world, PORT[0], folder, task, cfg), nprocs=world, join=True)
    return [torch.load(os.path.join(folder, f"{task}-w{world}-r{r}.pt")) for r in range(world)]


def relative(a, b, start, frobenius=False):
    """|a - b| over the size of the move from ``start`` (the updates are what is being compared): worst
    element, or per-tensor Frobenius."""
    if frobenius:
        return max(((a[n] - b[n]).norm() / (b[n] - start[n]).norm().clamp(min=1e-12)).item() for n in b)
    return max((a[n] - b[n]).abs().max().item() / max((b[n] - start[n]).abs().max().item(), 1e-12) for n in b)


def union(outs, key):
    merged = {}
    for out in outs:
        assert not set(merged) & set(out[key]), f"{key}: a parameter has two owners"
        merged.update(out[key])
    return merged


def compare_states(got, want, tol, label):
    assert set(got) == set(want), f"{label}: optimizer state keys differ"
    err = max(((got[n][k] - want[n][k]).abs().max() / want[n][k].abs().max().clamp(min=1e-12)).item()
              for n in want for k in want[n])
    assert err < tol, f"{label}: optimizer state relative error {err:.3g}"
    return err


def check_parity(world, folder, ref, start, cfg, tol, label, frobenius=False, norm_tol=1e-4):
    outs = launch(world, folder, "parity", cfg)
    owned = [n for out in outs for n in out["owned"]]
    assert sorted(owned) == sorted(start), "every parameter owned exactly once"
    for out in outs:
        assert all(torch.equal(out["params"][n], outs[0]["params"][n]) for n in start), "replicas differ"
        assert out["restored"], "swapped did not restore the live weights"
        assert relative(out["swapped"], ref["ema"], start, frobenius) < tol, "swapped replica != full EMA"
    err_w = relative(outs[0]["params"], ref["params"], start, frobenius)
    assert err_w < tol, f"{label}: weights {err_w:.3g}"
    err_e = relative(union(outs, "ema"), ref["ema"], start, frobenius)
    assert err_e < tol, f"EMA {err_e:.3g}"
    err_s = compare_states(union(outs, "states"), ref["states"], tol, label)
    norm_err = max(abs(a - b) / b for a, b in zip(outs[0]["norms"], ref["norms"]))
    assert norm_err < norm_tol and ref["norms"][0] > cfg["max_grad_norm"], "global norm / clip inactive"
    print(f"  ok  {label}: weights {err_w:.1e}, EMA {err_e:.1e}, optimizer states {err_s:.1e}, "
          f"norm {norm_err:.1e} ({'Frobenius ' if frobenius else ''}relative to the update); "
          f"replicas identical, swapped == EMA, restored")


def main():
    start = {n: p.detach().clone() for n, p in tiny_model().named_parameters()}
    with tempfile.TemporaryDirectory() as folder:
        for world in (1, 3):
            ref = reference(world, UPDATES, CFG)
            check_parity(world, folder, ref, start, CFG, 1e-3, f"world {world} zero == single process")

        ref = reference(3, UPDATES, CFG)
        outs = launch(3, folder, "local", CFG)
        assert abs(outs[0]["norms"][0] - ref["norms"][0]) / ref["norms"][0] < 1e-4, "local: logged norm not global"
        worst = max(x for out in outs for x in out["local_norms"])
        assert worst <= CFG["max_grad_norm"] * (1 + 1e-4), f"local clip: owner norm {worst:.3g}"
        print(f"  ok  zero.clip local: logged norm is global ({outs[0]['norms'][0]:.3g}), every owner's grads "
              f"<= {CFG['max_grad_norm']} (worst {worst:.3g})")

        low = {**CFG, "comm_dtype": torch.bfloat16, "muon": {"state_dtype": "bfloat16"}}
        # bf16 grads / momentum are ~4e-3 per element; Newton-Schulz amplifies that along small singular
        # directions to ~10% of the update (Frobenius), the same as Muon's own bf16 orthogonalization does to
        # an fp32 ulp. A sharding bug is still O(1).
        check_parity(3, folder, ref, start, low, 0.3, "world 3 bf16 grads + bf16 Muon momentum ~= reference",
                     frobenius=True, norm_tol=2e-2)

        outs = launch(3, folder, "resume", CFG)
        err = max(out["resume_err"] for out in outs)
        assert err == 0.0, f"resume differs from uninterrupted: {err:.3g}"
        print("  ok  DCP per-rank keys: resume at world 3 == uninterrupted (bit-exact)")

        ref2 = reference(3, 2, CFG)
        outs = launch(2, folder, "init_from", CFG)
        for out in outs:
            assert all(torch.equal(out["params"][n], outs[0]["params"][n]) for n in start), "replicas differ"
        err_w = relative(outs[0]["params"], ref2["params"], start)
        assert err_w < 1e-3, f"init_from weights {err_w:.3g}"
        err_s = compare_states(union(outs, "states"), ref2["states"], 1e-3, "init_from")
        print(f"  ok  world-3 checkpoint -> world 2 (new owners): weights {err_w:.1e}, owned states {err_s:.1e}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

"""check_fast_lora_parity.py — step-0 loss/grad parity: plain engine vs Pipeline.

One cached chunie batch, fixed noise and timestep, the real 12B base with a
rank-8 adapter whose B is randomized (so A gets nonzero gradients too). Runs:

  plain   mask=None + cuDNN / math
  plain   masked + cuDNN      (all-valid mask, 256-padded sequence)
  plain   compiled blocks     (max-autotune-no-cudagraphs; flash = the fast
          batch-1 path, math = closest to fp32)
  pipeline resident single stage, grad checkpointing (train_concept_lora's
           original engine)

and compares each against an fp32 ground truth (fp32 weights, math SDPA, no
autocast). Every bf16 engine is ~1x off fp32 on the adapters upstream of
all blocks, so the gate is "no worse than the pipeline engine". PARITY_SEED
picks the adapter init; PARITY_DUMP=path saves all gradients.
GPU, ~1 minute. Needs the cache built by train_concept_lora.py.

    CUDA_VISIBLE_DEVICES=1 uv run python krea2/tools/check_fast_lora_parity.py \
        krea2/configs/train_fast_lora_chunie.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from ramtorch import Pipeline

from krea2.model.chunks import balance_chunks_by_bytes, build_dit_chunks, set_dit_grad_ckpt
from krea2.model.configs import MMDIT_CONFIGS
from krea2.model.lora import inject_lora
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.sampling import prepare
from krea2.train_utils import _pin_sdpa_backends
from utils.ramtorch_helpers import (
    allow_tuple_infer,
    drop_frozen_params,
    flush_grads,
    set_resident_out_no_grad,
)


def main():
    cfg = json.load(open(sys.argv[1]))
    device, dtype = "cuda:0", torch.bfloat16
    _pin_sdpa_backends()
    manifest = json.load(open(os.path.join(cfg["cache_path"], "manifest.json")))
    img = manifest["images"][0]
    x0 = load_file(img["latent_path"])["latent"].to(device, dtype)
    d = load_file(img["bands"]["long_caption"]["path"])
    txt = d["hidden"][: int(d["valid_len"][0])][None].to(device, dtype)

    dit_cfg = MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")]
    with torch.device("meta"):
        dit = SingleStreamDiT(dit_cfg)
    dit.load_state_dict(load_file(cfg["mmdit_checkpoint"]), strict=True, assign=True)
    seed = int(os.environ.get("PARITY_SEED", 0))
    torch.manual_seed(seed)               # lora_A init draws from the global RNG
    inject_lora(dit, rank=cfg["lora_rank"], alpha=cfg["lora_alpha"],
                exclude_prefixes=("tagembed", *cfg["lora_exclude_prefixes"]))
    for name, p in dit.named_parameters():
        p.requires_grad_(name.endswith((".lora_A", ".lora_B")))
    gen = torch.Generator().manual_seed(seed)
    for name, p in dit.named_parameters():
        if name.endswith(".lora_B"):
            p.data.normal_(0, 1e-3, generator=gen)
    dit = dit.float().to(device)          # fp32 first: the ground-truth pass
    adapters = [(n, p) for n, p in dit.named_parameters() if p.requires_grad]

    torch.manual_seed(0)
    t = torch.tensor([0.7], device=device)
    noise = torch.randn_like(x0)
    t4 = t[:, None, None, None].to(dtype)
    x_t = (1 - t4) * x0 + t4 * noise
    v_target = rearrange(noise - x0, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    L = txt.shape[1]
    tok, pos, mask = prepare(x_t, L, 2, torch.ones(1, L, dtype=torch.bool, device=device))

    def grads():
        out = {n: p.grad.detach().clone() for n, p in adapters}
        for _, p in adapters:
            p.grad = None
        return out

    # fp32 weights, fp32 math attention, no autocast. Block 0 has attention
    # logits ~3e4, so the bf16 backward there is badly conditioned and two
    # bf16 kernels can disagree with each other; judge both against this.
    dit.grad_ckpt = True                  # fp32 activations would not fit
    with sdpa_kernel(SDPBackend.MATH):
        out = dit(tok.float(), txt.float(), t, pos, None)
    fp32_loss = F.mse_loss(out, v_target.float())
    fp32_loss.backward()
    fp32_loss, truth = fp32_loss.item(), grads()
    del out
    dit.grad_ckpt = False
    dit = dit.to(dtype)
    for name, p in adapters:
        p.data = p.data.float()
    torch.cuda.empty_cache()

    def plain(use_mask, backend):
        with sdpa_kernel(backend), torch.autocast("cuda", dtype=dtype):
            out = dit(tok, txt, t, pos, mask if use_mask else None)
        loss = F.mse_loss(out.float(), v_target.float())
        loss.backward()
        return loss.item(), grads()

    results = {
        "plain unmasked cuDNN": plain(False, SDPBackend.CUDNN_ATTENTION),
        "plain masked cuDNN": plain(True, SDPBackend.CUDNN_ATTENTION),
        "plain unmasked math": plain(False, SDPBackend.MATH),
    }

    from krea2.compile import compile_plain_dit_blocks
    import krea2.model.mmdit as _mmdit
    eager_forwards = {id(b): b.forward for b in dit.modules() if hasattr(b, "attn")}
    compile_plain_dit_blocks(dit, {"mode": "max-autotune-no-cudagraphs", "dynamic": True})
    for label, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("math", SDPBackend.MATH)):
        torch._dynamo.reset()
        _mmdit.set_sdpa_ctx(True, backend)        # compile ignores the outer context
        results[f"plain compiled {label}"] = plain(False, backend)
    _mmdit.set_sdpa_ctx(False)
    for b in dit.modules():
        if id(b) in eager_forwards:
            b.forward = eager_forwards[id(b)]
            del b._krea_compile_options

    chunks = build_dit_chunks(dit, blocks_per_chunk=1)
    counts = balance_chunks_by_bytes(chunks, 1)
    set_dit_grad_ckpt(chunks, True, counts)
    pipe = Pipeline(chunk_modules=chunks, chunks_per_stage=counts, devices=[device],
                    autocast=dtype, offload=False)
    set_resident_out_no_grad(pipe, (3,))
    allow_tuple_infer(pipe)
    drop_frozen_params(pipe)
    chunks[-1].set_seq(L, tok.shape[1], taglen=0)
    res = pipe.step(((tok, txt, t, pos, mask),), targets=v_target,
                    schedule="staggered_1b1f", n_microbatches=1,
                    loss_fn=lambda o, y: F.mse_loss(o.float(), y.float()))
    flush_grads(pipe, 1)
    results["pipeline cuDNN ckpt"] = (float(torch.as_tensor(res.loss).item()),
                                      {n: p.grad.detach().clone() for n, p in adapters})
    pipe.close()

    if os.environ.get("PARITY_DUMP"):
        torch.save({"fp32_loss": fp32_loss, "truth": truth, "results": results},
                   os.environ["PARITY_DUMP"])

    def rel_err(g, ref, names):
        num = sum(((g[n] - ref[n]).float() ** 2).sum() for n in names)
        den = sum((ref[n].float() ** 2).sum() for n in names)
        return (num / den).sqrt().item()

    early = [n for n in truth if n.startswith(("txtfusion.", "txtmlp.", "first.", "tmlp.",
                                               "tproj.", "blocks.0."))]
    late = [n for n in truth if n not in early]
    print(f"\nfp32 ground truth loss {fp32_loss:.6f}  (early = text path + block 0, "
          f"upstream of block 0's ~3e4-logit attention; late = blocks 1..27 + head)")
    errs = {}
    for name, (loss, g) in results.items():
        errs[name] = rel_err(g, truth, truth)
        print(f"  {name:22s} loss {loss:.6f} (d {loss - fp32_loss:+.2e})  grad rel err "
              f"all {errs[name]:.2e}  early {rel_err(g, truth, early):.2e}  "
              f"late {rel_err(g, truth, late):.2e}")
    # bf16 autocast alone puts every engine ~1x off fp32 on the upstream
    # adapters (tmlp/first/txtfusion sum gradients over all 28 blocks), so
    # the gate is relative: no engine materially worse than the pipeline
    # engine the existing runs used. Eager math is the no-compile fallback
    # and rounds block 0's ~3e4 logits to bf16 (step 128); it only warns.
    base_err = errs["pipeline cuDNN ckpt"]
    base_d = abs(results["pipeline cuDNN ckpt"][0] - fp32_loss)
    ok = True
    for k, e in errs.items():
        if e > 1.25 * base_err or abs(results[k][0] - fp32_loss) > 1.5 * base_d + 1e-5:
            print(f"  {'WARN' if k == 'plain unmasked math' else 'FAIL'}: {k} worse than pipeline")
            ok &= k == "plain unmasked math"
    print("\nPARITY PASSED" if ok else "\nPARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

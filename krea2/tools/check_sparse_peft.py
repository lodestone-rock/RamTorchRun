"""CPU/CUDA regression checks for fixed-coordinate PEFT and RamTorch capture."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import copy
import tempfile
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file
from ramtorch import Pipeline
from krea2.model import sparse_peft as sp
from utils.ramtorch_helpers import flush_grads


def check(device, offload):
    torch.manual_seed(42)
    base = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 8))
    model = copy.deepcopy(base)
    sp.inject(model, n_slots=1, rank=2)
    x, target = torch.randn(4, 16), torch.randn(4, 8)
    # Saliency must be abs(mean(microbatch gradients)), not mean(abs(g)).
    ref = copy.deepcopy(base)
    expected = torch.autograd.grad((ref(x)-target).square().mean(), [ref[0].weight, ref[2].weight])
    for xb, yb in zip(x.chunk(2), target.chunk(2)):
        ((model(xb)-yb).square().mean()).backward()
    opt = sp.SparseAdam(model, lr=.003, warmup=0, n_microbatches=2, max_grad_norm=1e9)
    opt.collect()
    for (_, m), g in zip(sp.modules(model), expected):
        torch.testing.assert_close(m.store.score, g.flatten().abs())
    with tempfile.TemporaryDirectory() as tmp:
        mask = Path(tmp)/'mask.safetensors'
        selected = opt.state_dict()
        assert sum(v.numel() for v in selected.values()) == model._sparse_budget
        save_file(selected, mask)
        model = copy.deepcopy(base)
        sp.inject(model, n_slots=1, rank=2, mask_path=str(mask))
        pipe = Pipeline(chunk_modules=list(model.children()), chunks_per_stage=[3],
                        devices=[device], offload=offload, offload_window=2,
                        offload_keep_activations='checkpoint', offload_grad_accum='cpu')
        sp.release_accumulators(pipe)
        opt = sp.SparseAdam(model, lr=.003, warmup=0, n_microbatches=2, max_grad_norm=1e9)
        ref = copy.deepcopy(base).to(device)
        ref_params = [ref[0].weight, ref[2].weight]
        denseopt = torch.optim.AdamW(ref_params, lr=.003, betas=(.9,.95), weight_decay=0)
        try:
            for cycle in range(3):
                nested = tuple((xb.to(device),) for xb in x.chunk(2))
                pipe.step(nested, targets=target.to(device), n_microbatches=2,
                          loss_fn=lambda a,b: (a-b).square().mean())
                opt.collect()
                denseopt.zero_grad()
                ((ref(x.to(device))-target.to(device)).square().mean()).backward()
                for (_, m), p, compact in zip(sp.modules(model), ref_params, opt.params):
                    idx = m.store.indices.to(device)
                    torch.testing.assert_close(compact.grad, p.grad.flatten()[idx].cpu(), atol=1e-7, rtol=1e-5)
                    filtered = torch.zeros_like(p.grad).flatten()
                    filtered[idx] = p.grad.flatten()[idx]
                    p.grad.copy_(filtered.view_as(p))
                denseopt.step()
                opt.step([0])
                if offload:
                    flush_grads(pipe, 2)
                for (name,m), p in zip(sp.modules(model), ref_params):
                    torch.testing.assert_close(m.weight.cpu(), p.cpu(), atol=1e-6, rtol=1e-5)
                    frozen = torch.ones(m.weight.numel(), dtype=torch.bool)
                    frozen[m.store.indices] = False
                    assert torch.equal(m.weight.cpu().flatten()[frozen], base.get_submodule(name).weight.flatten()[frozen])
                    assert m.weight.grad is None
            delta = Path(tmp)/'delta.safetensors'
            save_file(opt.state_dict(), delta)
            restored = copy.deepcopy(base)
            sp.apply_adapter(restored, str(delta))
            for name,m in sp.modules(model):
                torch.testing.assert_close(restored.get_submodule(name).weight, m.weight.cpu())
            # Compact optimizer restore: next update agrees including moments.
            resumed = copy.deepcopy(base)
            sp.inject(resumed, n_slots=1, rank=2, mask_path=str(mask))
            other = sp.SparseAdam(resumed, lr=.003, warmup=0, n_microbatches=2, max_grad_norm=1e9)
            other.opt.load_state_dict(copy.deepcopy(opt.opt.state_dict()))
            other.slot_steps = opt.slot_step_counts()
            for p,q in zip(other.params,opt.params):
                p.data.copy_(q)
                p.grad = torch.full_like(p,.01)
                q.grad = torch.full_like(q,.01)
            other.step([0]); opt.step([0])
            for p,q in zip(other.params,opt.params):
                assert torch.equal(p,q)
        finally:
            pipe.close()
    print(f'PASS {device} offload={offload}: saliency, gradient averaging, masked Adam, frozen weights, delta roundtrip')


def check_k2():
    from krea2.tools.check_lora_bank import TINY, make_inputs
    from krea2.model.mmdit import SingleStreamDiT, set_sdpa_ctx
    from krea2.model.chunks import build_dit_chunks, set_dit_grad_ckpt
    from utils.ramtorch_helpers import set_resident_out_no_grad
    set_sdpa_ctx(False)
    torch.manual_seed(12)
    base = SingleStreamDiT(TINY)
    inputs, imglen = make_inputs(TINY, 2)
    target = torch.randn(2, imglen, TINY.channels * TINY.patch**2)
    ref = copy.deepcopy(base)
    ref_out = ref(*inputs)
    ((ref_out-target)**2).mean().backward()
    for checkpoint in (False, True):
        model = copy.deepcopy(base)
        sp.inject(model, n_slots=1, rank=4)
        chunks = build_dit_chunks(model)
        chunks[-1].set_seq(inputs[1].shape[1], imglen)
        if checkpoint:
            set_dit_grad_ckpt(chunks, 1., [len(chunks)])
        pipe = Pipeline(chunk_modules=chunks, chunks_per_stage=[len(chunks)], devices=['cpu'])
        set_resident_out_no_grad(pipe, (3,))
        sp.release_accumulators(pipe)
        nested = tuple(tuple(t[i:i+1] for t in inputs) for i in range(2))
        try:
            pipe.step(nested, targets=target, n_microbatches=2, loss_fn=lambda a,b: (a-b).square().mean())
            for name,m in sp.modules(model):
                assert m.store.calls == 2
                torch.testing.assert_close(m.store.grad/2, ref.get_submodule(name).weight.grad.flatten(), atol=1e-6, rtol=1e-4)
        finally:
            pipe.close()
        print(f'PASS tiny K2 every captured gradient vs monolithic, checkpoint={checkpoint}')


def check_bf16_master():
    layer = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    layer.weight.data.fill_(1)
    model = nn.Sequential(sp.SparseLinear(layer, torch.tensor([0, 2])))
    opt = sp.SparseAdam(model, lr=1e-4, warmup=0, n_microbatches=1, max_grad_norm=1)
    for _ in range(100):
        model[0].store.add(torch.ones_like(layer.weight))
        opt.collect(); opt.step([0])
    assert model[0].weight.flatten()[0] < 1
    assert torch.equal(model[0].weight.flatten()[3:], torch.ones_like(layer.weight.flatten()[3:]))
    print('PASS BF16 compute with FP32 compact master retains sub-ULP updates')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    torch.set_num_threads(2)
    for offload in (False, True):
        check(args.device, offload)
    if args.device == 'cpu':
        check_k2()
        check_bf16_master()
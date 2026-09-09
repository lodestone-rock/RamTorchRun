"""Inflight sparse saliency vs independent dense implementation of filtering."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import copy
import tempfile
from pathlib import Path
import torch
from torch import nn
from ramtorch import Pipeline
from safetensors.torch import save_file
from krea2.model import sparse_peft as sp


def dense_band(g, k):
    # Stable descending sort gives coordinate-order ties, independent of topk.
    ids = torch.argsort(g.abs(), descending=True, stable=True)[:k]
    result = torch.zeros_like(g)
    result[ids] = g[ids]
    return result


def store_test(device):
    torch.manual_seed(7)
    store = sp.InflightStore(64, 9)
    reference = torch.zeros(64, device=device)
    raw_score = torch.zeros_like(reference)
    for step in range(10):
        grads = torch.randn(4, 64, device=device)
        if step == 0:
            grads.zero_()  # exact ties, exact budget
        if step == 1:
            grads[1] = -grads[0]  # signed microbatch cancellation
        filtered = torch.zeros_like(reference)
        for g in grads:
            store.add(g)
            filtered += dense_band(g, 9) / 4
        reference += dense_band(filtered, 9).abs()
        raw_score += grads.mean(0).abs()
        store.finish_step(4)
        torch.testing.assert_close(store.score.to_dense(), reference)
        assert store.grad is None and store.calls == 0
        assert store.score._nnz() <= min(64, (step + 1) * 9)
    assert not torch.allclose(reference, raw_score)
    # Full-capacity control recovers the exact dense batch-mean saliency.
    exact = sp.InflightStore(64,64)
    grads=torch.randn(4,64,device=device)
    for g in grads:
        exact.add(g)
    exact.finish_step(4)
    torch.testing.assert_close(exact.score.to_dense(),grads.mean(0).abs())
    print(f'PASS {device}: ten overlapping sparse steps, ties, cancellation, storage bound, explicit approximation')


def pipeline_test(device, offload):
    torch.manual_seed(10)
    base = nn.Sequential(nn.Linear(16,16),nn.SiLU(),nn.Linear(16,8))
    model = copy.deepcopy(base)
    sp.inject(model,n_slots=1,rank=2,calibration_mode='inflight',candidate_multiplier=2)
    pipe = Pipeline(chunk_modules=list(model.children()),chunks_per_stage=[3],devices=[device],
                    offload=offload,offload_window=2,offload_keep_activations='checkpoint')
    sp.release_accumulators(pipe)
    opt = sp.SparseAdam(model,lr=1e-4,warmup=0,n_microbatches=2,max_grad_norm=1)
    reference = copy.deepcopy(base).to(device)
    expected = {name:torch.zeros(m.weight.numel(),device=device) for name,m in sp.modules(model)}
    try:
        for step in range(3):
            x,y=torch.randn(4,16,device=device),torch.randn(4,8,device=device)
            step_grad={name:torch.zeros_like(g) for name,g in expected.items()}
            for xb,yb in zip(x.chunk(2),y.chunk(2)):
                reference.zero_grad()
                (reference(xb)-yb).square().mean().backward()
                for name,m in sp.modules(model):
                    g=reference.get_submodule(name).weight.grad.flatten()
                    step_grad[name] += dense_band(g,m.store.capacity)/2
            for name,m in sp.modules(model):
                expected[name] += dense_band(step_grad[name],m.store.capacity).abs()
            pipe.step(tuple((xb,) for xb in x.chunk(2)),targets=y,n_microbatches=2,
                      loss_fn=lambda a,b:(a-b).square().mean())
            opt.collect();opt.step([0])
            for name,m in sp.modules(model):
                torch.testing.assert_close(m.store.score.to_dense(),expected[name],atol=1e-6,rtol=1e-5)
                assert m.weight.grad is None
        mask = opt.state_dict()
        combined=torch.cat(list(expected.values()))
        chosen=torch.argsort(combined,descending=True,stable=True)[:model._sparse_budget]
        wanted=torch.zeros_like(combined,dtype=torch.bool);wanted[chosen]=True
        offset=0
        for name,m in sp.modules(model):
            ids=wanted[offset:offset+m.weight.numel()].nonzero().flatten().cpu()
            assert torch.equal(mask[name+'.indices'],ids)
            offset+=m.weight.numel()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mask.safetensors';save_file(mask,path)
            restored=copy.deepcopy(base)
            sp.inject(restored,n_slots=1,rank=2,mask_path=str(path))
        print(f'PASS {device} offload={offload}: actual capture, sparse step scores, global budget, frozen base, saved mask')
    finally:
        pipe.close()


def tiny_k2_test():
    from krea2.tools.check_lora_bank import TINY, make_inputs
    from krea2.model.mmdit import SingleStreamDiT, set_sdpa_ctx
    from krea2.model.chunks import build_dit_chunks, set_dit_grad_ckpt
    from utils.ramtorch_helpers import set_resident_out_no_grad
    set_sdpa_ctx(False)
    torch.manual_seed(19)
    model=SingleStreamDiT(TINY)
    sp.inject(model,n_slots=1,rank=4,calibration_mode='inflight')
    reference=copy.deepcopy(model)
    inputs,imglen=make_inputs(TINY,2)
    target=torch.randn(2,imglen,TINY.channels*TINY.patch**2)
    for i in range(2):
        (reference(*(x[i:i+1] for x in inputs))-target[i:i+1]).square().mean().backward()
    chunks=build_dit_chunks(model)
    chunks[-1].set_seq(inputs[1].shape[1],imglen)
    set_dit_grad_ckpt(chunks,1.,[len(chunks)])
    pipe=Pipeline(chunk_modules=chunks,chunks_per_stage=[len(chunks)],devices=['cpu'])
    set_resident_out_no_grad(pipe,(3,));sp.release_accumulators(pipe)
    try:
        pipe.step(tuple(tuple(x[i:i+1] for x in inputs) for i in range(2)),targets=target,
                  n_microbatches=2,loss_fn=lambda a,b:(a-b).square().mean())
        for (name,m),(_,ref) in zip(sp.modules(model),sp.modules(reference)):
            assert m.store.calls==ref.store.calls==2
            # Floating batch order may alter boundary topk ties; compare packets
            # numerically, with tiny K2's identical CPU execution as the control.
            torch.testing.assert_close(m.store.grad.to_dense(),ref.store.grad.to_dense(),atol=1e-6,rtol=1e-4)
        print('PASS tiny K2 inflight packet parity through checkpointed chunks')
    finally:
        pipe.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');args=ap.parse_args()
    torch.set_num_threads(2)
    store_test(args.device)
    for offload in (False,True):
        pipeline_test(args.device,offload)
    if args.device=='cpu':
        tiny_k2_test()
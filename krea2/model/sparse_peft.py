"""Fixed salient-coordinate PEFT, initially restricted to one artist.

Capture stops parameter gradients before RamTorch's dense accumulation. Dense
weight-gradient GEMMs still run; only persistent gradients/moments are compact.
"""
from __future__ import annotations

import numpy as np
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file


def select_top(indices, values, count):
    """Top magnitude, deterministic coordinate-order ties; compact tensors only."""
    if values.numel() <= count:
        return indices, values
    magnitude = values.abs()
    threshold = magnitude.topk(count, sorted=False).values.min()
    above = (magnitude > threshold).nonzero().flatten()
    ties = (magnitude == threshold).nonzero().flatten()
    # Inputs are coordinate-sorted. Taking the first ties is reproducible.
    chosen = torch.cat((above, ties[:count - above.numel()])).sort().values
    return indices[chosen], values[chosen]


class InflightStore:
    """GPU top-tail capture -> signed step sum -> absolute sparse score sum.

No full-model dense accumulators. Each backward is filtered BEFORE averaging,
so this is explicitly an approximation to dense mean-absolute-gradient saliency.
Missing coordinates contribute zero. Historical scores are NOT pruned until the
final global selection; overlapping coordinates are summed by COO coalescing.
"""
    def __init__(self, size, capacity):
        self.size, self.capacity = size, min(size, capacity)
        self.indices = None
        self.grad = None
        self.score = None
        self.calls = 0
        self.steps = 0

    def packet(self, indices, values):
        return torch.sparse_coo_tensor(indices[None], values, (self.size,),
                                      device=values.device).coalesce()

    def add(self, grad):
        flat = grad.detach().flatten()
        # Only this map's gradient is transiently dense. Compute topk on GPU;
        # float conversion is restricted to selected values, not the whole map.
        mag = flat.abs()
        threshold = mag.topk(self.capacity, sorted=False).values.min()
        above = (mag > threshold).nonzero().flatten()
        ties = (mag == threshold).nonzero().flatten()
        idx = torch.cat((above, ties[:self.capacity - above.numel()])).sort().values
        value = flat[idx].float()
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite inflight gradient')
        packet = self.packet(idx, value)
        self.grad = packet if self.grad is None else (self.grad + packet).coalesce()
        self.calls += 1

    def finish_step(self, microbatches):
        if self.calls != microbatches or self.grad is None:
            raise RuntimeError(f'Inflight capture count {self.calls}, expected {microbatches}')
        g = self.grad.coalesce()
        idx, values = select_top(g.indices()[0], g.values() / microbatches, self.capacity)
        packet = self.packet(idx, values.abs())
        self.score = packet if self.score is None else (self.score + packet).coalesce()
        self.steps += 1
        self.clear()

    def clear(self):
        self.grad = None
        self.calls = 0


def inflight_mask(model):
    """Final global top-k of the sparse average, on the capture GPU."""
    mods = modules(model)
    steps = {m.store.steps for _, m in mods}
    if len(steps) != 1 or not next(iter(steps)):
        raise ValueError('Inconsistent/empty inflight calibration')
    nsteps = next(iter(steps))
    scores = [m.store.score.coalesce() for _, m in mods]
    if len({s.device for s in scores}) != 1:
        raise ValueError('Inflight final selection currently requires one calibration device')
    values = torch.cat([s.values() / nsteps for s in scores])
    k = model._sparse_budget
    if values.numel() < k:
        raise ValueError('Candidate union smaller than final budget; increase candidate multiplier')
    threshold = values.topk(k, sorted=False).values.min()
    del values
    selected = [s.indices()[0][s.values() / nsteps > threshold] for s in scores]
    remaining = k - sum(idx.numel() for idx in selected)
    for i, s in enumerate(scores):
        if remaining:
            ties = s.indices()[0][s.values() / nsteps == threshold][:remaining]
            selected[i] = torch.cat((selected[i], ties)).sort().values
            remaining -= ties.numel()
    assert remaining == 0
    print(f'[Inflight] GPU global selection: {sum(s._nnz() for s in scores):,} candidates '
          f'-> {k:,} coordinates after {nsteps} steps; threshold={float(threshold):.8g}', flush=True)
    return {name + '.indices': idx.cpu() for (name, _), idx in zip(mods, selected)}


class Store:
    def __init__(self, indices=None):
        self.indices = indices
        self.grad = None
        self.score = None
        self.calls = 0
        self.device_indices = {}

    def add(self, grad):
        flat = grad.detach().flatten()
        if self.indices is not None:
            if grad.device not in self.device_indices:
                self.device_indices[grad.device] = self.indices.to(grad.device)
            idx = self.device_indices[grad.device]
            flat = flat.index_select(0, idx)
        value = flat.to(device='cpu', dtype=torch.float32)  # blocking, worker-owned
        if self.grad is None:
            self.grad = value.clone() if flat.device.type == 'cpu' else value
        else:
            # CPU NumPy ufunc avoids an OpenMP parallel region inside the CUDA
            # autograd worker for each dense calibration gradient.
            np.add(self.grad.numpy(), value.numpy(), out=self.grad.numpy())
        self.calls += 1

    def clear(self):
        self.grad = None
        self.calls = 0


class Capture(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, store):
        ctx.store = store
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx, grad):
        ctx.store.add(grad)
        return None, None


class SparseLinear(nn.Module):
    def __init__(self, layer, indices):
        super().__init__()
        self.in_features, self.out_features = layer.in_features, layer.out_features
        self.weight = nn.Parameter(layer.weight.detach())
        self.register_buffer('bias', None if layer.bias is None else layer.bias.detach())
        self.store = Store(indices)

    def forward(self, x):
        weight = Capture.apply(self.weight, self.store) if torch.is_grad_enabled() else self.weight
        return F.linear(x, weight, self.bias)


def modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, SparseLinear)]


def frozen_digest(module):
    values = module.weight.detach().cpu().flatten().clone()
    if module.store.indices is not None:
        values[module.store.indices] = 0
    return hashlib.sha256(memoryview(values.view(torch.uint8).numpy())).hexdigest()


def inject(model, *, n_slots, rank, alpha=None, exclude_prefixes=(), exclude_patterns=(), mask_path=None,
           calibration_mode='dense', candidate_multiplier=2):
    if n_slots != 1:
        raise ValueError('Sparse PEFT currently supports exactly one slot; shared updates would mix artists')
    masks = load_file(mask_path) if mask_path else None
    if calibration_mode not in ('dense', 'inflight') or candidate_multiplier < 1:
        raise ValueError('Invalid calibration mode or candidate multiplier')
    selected_names = []
    budget = 0
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, nn.Linear) or rank >= min(layer.in_features, layer.out_features):
            continue
        if any(name.startswith(p) for p in exclude_prefixes) or any(p in name for p in exclude_patterns):
            continue
        idx = None if masks is None else masks[name + '.indices']
        if idx is not None:
            if idx.dtype != torch.int64 or idx.ndim != 1 or (idx.numel() and
                    (idx.min() < 0 or idx.max() >= layer.weight.numel() or not torch.all(idx[1:] > idx[:-1]))):
                raise ValueError(f'Invalid coordinate mask: {name}')
        parent, _, leaf = name.rpartition('.')
        replacement = SparseLinear(layer, idx)
        if masks is None and calibration_mode == 'inflight':
            import math
            replacement.store = InflightStore(layer.weight.numel(),
                math.ceil(candidate_multiplier * rank * (layer.in_features + layer.out_features)))
        setattr(model.get_submodule(parent), leaf, replacement)
        selected_names.append(name + '.indices')
        budget += rank * (layer.in_features + layer.out_features)
    if masks is not None and (set(masks) != set(selected_names) or sum(x.numel() for x in masks.values()) != budget):
        raise ValueError('Mask coverage/budget differs from matched LoRA')
    # Norms, biases and excluded maps are genuinely frozen buffers, not merely
    # omitted from the optimizer (avoids dense unused gradient accumulators).
    for layer in model.modules():
        for name, p in list(layer.named_parameters(recurse=False)):
            if isinstance(layer, SparseLinear) and name == 'weight':
                continue
            del layer._parameters[name]
            layer.register_buffer(name, p.detach())
    model._sparse_budget = budget
    print(f'[Sparse] {len(selected_names)} maps, matched budget {budget:,}, calibration={masks is None}')


def budget(model):
    count = model._sparse_budget
    return dict(modules=len(modules(model)), n_slots=1, params_per_slot=count, params_total=count, bytes=count * 4)


def set_active(model, slots):
    if list(slots) != [0]:
        raise ValueError('Sparse PEFT requires active slot [0]')


def release_accumulators(pipe):
    """Only captured weights: retain autograd inputs, discard unused buffers."""
    for stage in pipe.stages:
        if hasattr(stage, 'engine'):
            for state in stage.engine._state:
                state.grad_acc.clear()
        else:
            stage.grad_acc.clear()


class SparseAdam:
    def __init__(self, model, *, lr, warmup, n_microbatches, max_grad_norm, **kwargs):
        self.model, self.lr, self.warmup = model, lr, warmup
        self.n_microbatches, self.max_grad_norm = n_microbatches, max_grad_norm
        self.slot_steps, self.offloaded = [0], True
        self.calibration = modules(model)[0][1].store.indices is None
        self.params, self.original = [], []
        self.frozen_hashes = {name: frozen_digest(m) for name, m in modules(model)}
        if not self.calibration:
            for _, m in modules(model):
                original = m.weight.detach().flatten()[m.store.indices.to(m.weight.device)].float().cpu()
                self.original.append(original)
                self.params.append(nn.Parameter(original.clone()))
            self.opt = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.95), weight_decay=0)

    def lr_for(self, slot):
        return self.lr * (1 if not self.warmup else
                          1e-5 + (1 - 1e-5) * min(self.slot_steps[0], self.warmup) / self.warmup)

    def slot_step_counts(self):
        return list(self.slot_steps)

    def collect(self):
        for i, (_, m) in enumerate(modules(self.model)):
            s = m.store
            if isinstance(s, InflightStore):
                s.finish_step(self.n_microbatches)
                continue
            if s.calls != self.n_microbatches or s.grad is None:
                raise RuntimeError(f'Capture count {s.calls}, expected {self.n_microbatches}')
            if not torch.isfinite(s.grad).all():
                raise FloatingPointError('Nonfinite captured gradient')
            g = s.grad.div_(self.n_microbatches)
            if self.calibration:
                g.abs_()
                if s.score is None:
                    s.score = g
                else:
                    s.score.add_(g)
            else:
                self.params[i].grad = g
            s.clear()
        norm = torch.tensor(0.) if self.calibration else torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        return torch.tensor([float(norm)])

    @torch.no_grad()
    def step(self, slots):
        set_active(self.model, slots)
        if not self.calibration:
            for group in self.opt.param_groups:
                group['lr'] = self.lr_for(0)
            self.opt.step()
            for (_, m), p in zip(modules(self.model), self.params):
                m.weight.flatten().index_copy_(0, m.store.indices.to(m.weight.device), p.to(m.weight))
            self.opt.zero_grad(set_to_none=True)
        self.slot_steps[0] += 1

    def state_dict(self):
        for name, m in modules(self.model):
            if frozen_digest(m) != self.frozen_hashes[name]:
                raise AssertionError(f'Unselected weights changed: {name}')
        if self.calibration:
            if isinstance(modules(self.model)[0][1].store, InflightStore):
                return inflight_mask(self.model)
            # Exact global selection without a 12B-element int64 argsort.
            scores = np.concatenate([m.store.score.numpy() for _, m in modules(self.model)])
            k = self.model._sparse_budget
            scores.partition(len(scores) - k)
            threshold = float(scores[-k])
            del scores
            result, selected = {}, 0
            for name, m in modules(self.model):
                idx = (m.store.score > threshold).nonzero().flatten()
                result[name + '.indices'] = idx
                selected += len(idx)
            remaining = k - selected
            for name, m in modules(self.model):
                if remaining:
                    ties = (m.store.score == threshold).nonzero().flatten()[:remaining]
                    result[name + '.indices'] = torch.cat((result[name + '.indices'], ties)).sort().values
                    remaining -= len(ties)
            assert remaining == 0 and sum(x.numel() for x in result.values()) == k
            print(f'[Sparse] selected {k:,} coordinates; threshold={threshold:.8g}; '
                  f'per-map counts={ {name: len(idx) for name, idx in result.items()} }', flush=True)
            return result
        result = {}
        for (name, m), p, original in zip(modules(self.model), self.params, self.original):
            result[name + '.indices'] = m.store.indices
            result[name + '.delta'] = p.detach() - original
        return result


@torch.no_grad()
def apply_adapter(model, path):
    sd = load_file(path)
    names = {k.removesuffix('.indices') for k in sd if k.endswith('.indices')}
    if set(sd) != {n + suffix for n in names for suffix in ('.indices', '.delta')}:
        raise ValueError('Not a sparse delta checkpoint')
    for name in sorted(names):
        m = model.get_submodule(name)
        idx, delta = sd[name + '.indices'], sd[name + '.delta']
        if (idx.ndim != 1 or delta.shape != idx.shape or idx.dtype != torch.int64
                or not torch.isfinite(delta).all() or (idx.numel() and
                    (idx.min() < 0 or idx.max() >= m.weight.numel() or not torch.all(idx[1:] > idx[:-1])))):
            raise ValueError(f'Invalid sparse delta for {name}')
        flat = m.weight.flatten()
        idx = idx.to(flat.device)
        flat.index_copy_(0, idx, (flat[idx].float() + delta.to(flat.device)).to(flat.dtype))
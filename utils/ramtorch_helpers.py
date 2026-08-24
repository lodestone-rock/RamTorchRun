"""ramtorch_helpers.py — Plumbing shared by every chunk-based RamTorch script.

Model-agnostic glue for `Pipeline(chunk_modules=...)`: the couple of places
where a resident stage and a streamed (`OffloadStage`) one need different
handling, plus the frozen-module memory trick.
"""
from __future__ import annotations

import torch.nn as nn

from ramtorch.pipeline_offload import OffloadStage


def offload_stages(pipe) -> list:
    """The streamed stages of a pipeline (empty if it is fully resident)."""
    return [st for st in pipe.stages if isinstance(st, OffloadStage)]


class _TupleCallAdapter:
    """Let a streamed stage accept a relayed tuple during `Pipeline.infer`.

    The inference worker unpacks a relayed tuple into positional args
    (``stage.module(*x)``), but a streamed stage's ``module`` IS the
    `OffloadModel` engine, whose ``forward(x)`` takes ONE argument and unpacks
    tuples itself — so any multi-tensor relay raises "takes 2 positional
    arguments but 3 were given". `OffloadStage` sets ``self.module = engine``
    purely for this API and reads ``self.engine`` everywhere else, so wrapping
    it touches nothing but the inference call.
    """

    def __init__(self, engine):
        self.engine = engine

    def __call__(self, *args):
        return self.engine(args[0] if len(args) == 1 else args)

    def __getattr__(self, name):
        return getattr(self.engine, name)


def allow_tuple_infer(pipe):
    """Make `Pipeline.infer` work on streamed stages fed multi-tensor tuples."""
    for st in pipe.stages:
        if isinstance(st, OffloadStage) and not isinstance(st.module, _TupleCallAdapter):
            st.module = _TupleCallAdapter(st.engine)


def set_resident_out_no_grad(pipe, out_no_grad: tuple[int, ...]):
    """Declare which relayed tuple elements carry no gradient.

    Streamed stages read ``out_no_grad`` off the last chunk module, but
    resident stages are wrapped in RamTorch's private ``_ChunkSequential``,
    which has none of its own — so a float passthrough (a RoPE table, position
    ids) would be flagged grad-requiring at every stage boundary. `Stage`
    reads the attribute dynamically, so setting it post-construction works.
    """
    for st in pipe.stages:
        if not isinstance(st, OffloadStage):
            st.module.out_no_grad = out_no_grad


def set_resident_out_no_grad_per_stage(pipe, chunks, chunks_per_stage):
    """`set_resident_out_no_grad`, but read off each stage's LAST chunk.

    Needed when the relay tuple CHANGES SHAPE partway down the chunk list, so
    no single index set describes every boundary — radiance relays
    ``(x, img_px, t, mod, pe, attn_mask)`` through the transformer and
    ``(img_dct, nerf_cond, img_px, t)`` through the NeRF head. A stage's output
    is its last chunk's output, so that chunk's own declaration is the right
    one; streamed stages already read it there themselves.
    """
    idx = 0
    ends = []
    for cnt in chunks_per_stage:
        idx += cnt
        ends.append(chunks[idx - 1])
    for st, last in zip(pipe.stages, ends):
        if not isinstance(st, OffloadStage):
            st.module.out_no_grad = tuple(getattr(last, "out_no_grad", ()))


def flush_grads(pipe, n_microbatches: int):
    """Scale accumulated grads by 1/k and expose them as ``.grad``.

    Streamed stages go through the engine, which writes ``.grad`` on the CPU
    masters and then invalidates GPU weight residency so the next forward
    re-reads the post-optimizer masters.

    Resident stages deliberately skip ``result.flush_grads()``: that does
    ``p.grad = acc * scale``, allocating a SECOND full grad copy per stage
    (~14 GB/GPU for a 12B model in fp32). Aliasing ``p.grad`` to the
    accumulator and scaling in place is equivalent and free.
    """
    scale = 1.0 / n_microbatches
    for st in pipe.stages:
        if isinstance(st, OffloadStage):
            st.flush_grads(scale=scale)
        else:
            with st._lock:
                for p, acc in st.grad_acc.items():
                    acc.mul_(scale)          # mean of microbatch means
                    if p.grad is not acc:
                        p.grad = acc


def zero_grads(pipe):
    """Zero the accumulators in place — never set_to_none: for resident stages
    ``p.grad`` IS the accumulator the next backward adds into."""
    for st in pipe.stages:
        st.zero_grad_acc()


def prewarm_offload_staging(pipe):
    """Pre-allocate the ``grad_accum="cpu"`` D2H staging buffers with FULL
    parameter coverage.

    `OffloadModel._do_writeback` allocates a streamed chunk's pinned staging
    dict lazily, keyed by the FIRST backward's grad packet — and never extends
    it. With role-swapped LoRA (TDM) the first packet only carries one role's
    ``lora_*`` keys, so the other role's first update dies in the writeback
    thread with a KeyError. ``grad_acc`` always covers every parameter, so
    mirroring it is a strict superset of any packet. The extra pinned RAM over
    the lazy path is just the inactive role's adapters: the packet already
    contains the frozen-by-exclusion base grads (RamTorch computes grads for
    every param it streams). No-op for resident stages and "stream" accum.
    """
    import torch

    for st in offload_stages(pipe):
        eng = st.engine
        if getattr(eng, "grad_accum", None) != "cpu":
            continue
        for state in eng._state:
            if state.gpu_pinned or state.staging is not None:
                continue
            state.staging = {
                n: torch.empty(a.shape, dtype=a.dtype, device="cpu",
                               pin_memory=eng._cuda)
                for n, a in state.grad_acc.items()
            }


def drop_grad_accumulators(pipe):
    """Free the grad accumulators RamTorch allocates for a FROZEN module.

    Both `Stage` and `OffloadModel` allocate one accumulator per *parameter*,
    ignoring requires_grad — on a frozen 4B text encoder that is ~8 GB of host
    RAM (pinned, under grad_accum="stream"), and ~7 GB of VRAM on the biggest
    resident stage of a frozen 12.8B DiT. Only call this on modules that will
    never see a backward pass (`Pipeline.infer` only).
    """
    for st in pipe.stages:
        st.params = []
        if isinstance(st, OffloadStage):
            for state in st.engine._state:
                state.grad_acc = {}
        else:
            st.grad_acc = {}


class no_grad_accumulators:
    """Context manager that stops `Stage.__init__` allocating accumulators.

    `Stage.__init__` walks ``module.parameters()`` eagerly, so for a big frozen
    RESIDENT stage the allocation happens before there is any object to clean
    up. Hiding ``parameters()`` during construction leaves ``stage.params ==
    []``; ``module.to(device)`` uses ``_apply``, not ``parameters()``, so the
    weights still move. Streamed stages allocate inside the engine instead —
    use `drop_grad_accumulators` after construction for those.
    """

    def __enter__(self):
        self._orig = nn.Module.parameters
        nn.Module.parameters = lambda self, recurse=True: iter(())
        return self

    def __exit__(self, *exc):
        nn.Module.parameters = self._orig
        return False


# ---------------------------------------------------------------------------
# Optimizer: one per stage, so each GPU updates the weights it already streams
# ---------------------------------------------------------------------------

def chunk_params_by_stage(
    chunks: list[nn.Module],
    chunks_per_stage: list[int],
    devices: list[str],
    *,
    trainable_only: bool = True,
    exclude_ids: set[int] | None = None,
) -> list[tuple[str, list]]:
    """``[(device, params owned by the stage on it), ...]``.

    A chunk's parameters belong to exactly one stage, which is what lets the
    optimizer be sharded the same way the model is. Deduplicates by identity in
    case two chunks share a module.

    ``exclude_ids`` drops params by ``id()``. Under RamTorch nothing can be
    frozen with ``requires_grad_(False)``, so "frozen" always means "kept out of
    the optimizer" — this is how a caller passes that decision through.
    """
    exclude_ids = exclude_ids or set()
    groups, idx, seen = [], 0, set()
    for dev, cnt in zip(devices, chunks_per_stage):
        params = []
        for c in chunks[idx:idx + cnt]:
            for p in c.parameters():
                if (trainable_only and not p.requires_grad) or id(p) in seen:
                    continue
                if id(p) in exclude_ids:
                    continue
                seen.add(id(p))
                params.append(p)
        idx += cnt
        groups.append((dev, params))
    return groups


class MultiOptimizer:
    """A list of optimizers driven as one.

    Only the surface `train.py` uses (`step`, `zero_grad`, `param_groups`,
    state dict) — deliberately NOT a `torch.optim.Optimizer` subclass, since
    there is no single state to expose. Use `make_scheduler` to attach an LR
    schedule to every child.

    ``parallel`` steps the children in threads. This is the entire point when
    each child drives a different GPU: `OffloadAdamW.step` ends in a stream
    synchronize, so stepping sequentially would use one PCIe link at a time
    instead of all N. PyTorch's current stream is thread-local and each child
    only touches its own device, so the threads do not interfere.
    """

    def __init__(self, optimizers, parallel: bool = True):
        self.optimizers = list(optimizers)
        self._pool = None
        if parallel and len(self.optimizers) > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=len(self.optimizers))

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def step(self, closure=None):
        if self._pool is None:
            for o in self.optimizers:
                o.step()
            return
        futures = [self._pool.submit(o.step) for o in self.optimizers]
        for f in futures:
            f.result()          # re-raise worker exceptions

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, sds):
        for o, sd in zip(self.optimizers, sds):
            o.load_state_dict(sd)


class MultiScheduler:
    """The `MultiOptimizer` counterpart: one LR schedule per child optimizer.

    They are constructed identically and stepped together, so reading the LR
    off the first is the same as reading it off any of them.
    """

    def __init__(self, schedulers):
        self.schedulers = list(schedulers)

    def step(self, *a, **kw):
        for s in self.schedulers:
            s.step(*a, **kw)

    def get_last_lr(self):
        return self.schedulers[0].get_last_lr()


def make_scheduler(opt, factory):
    """``factory(optimizer) -> LRScheduler``, applied per child if sharded.

    `torch.optim.lr_scheduler` type-checks for a real `Optimizer`, so a
    `MultiOptimizer` cannot be handed to one directly.
    """
    if isinstance(opt, MultiOptimizer):
        return MultiScheduler([factory(o) for o in opt.optimizers])
    return factory(opt)


def build_offload_adamw(
    chunks: list[nn.Module],
    chunks_per_stage: list[int],
    devices: list[str],
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    bucket_mb: float = 32.0,
    window: int = 2,
    state_dtype=None,
    stochastic_rounding: bool = False,
    exclude_ids: set[int] | None = None,
) -> MultiOptimizer:
    """AdamW whose state lives in host RAM and whose math runs on the GPUs.

    `ramtorch.offload_optimizer.OffloadAdamW` streams (param, grad, exp_avg,
    exp_avg_sq) through a GPU window, so the update is bounded by PCIe rather
    than by a host pass over the state. Its own docstring argues the fused CPU
    kernel wins because DDR beats PCIe — which holds for ONE optimizer on ONE
    link. Here it is sharded per stage: each GPU updates only the chunks it
    owns, over its own link, so the aggregate bandwidth is N links against the
    host's single DDR bus. That is what flips the comparison on a 12.8B model
    spread over 4 GPUs.

    Params that are already GPU-resident (a non-offloaded stage) skip the
    window and get a direct on-device foreach update, so mixed pipelines work.

    ``state_dtype=torch.bfloat16`` halves both the pinned state (51 GB instead
    of 102 GB for 12.8B params) and the PCIe traffic; pair it with
    ``stochastic_rounding=True`` so repeated round-to-nearest does not bias the
    small updates.

    ``exclude_ids`` is forwarded to `chunk_params_by_stage` — params to keep out
    of the optimizer despite ``requires_grad=True`` (frozen-by-exclusion).
    """
    try:
        from ramtorch.offload_optimizer import OffloadAdamW
    except ImportError as e:  # private module — surface it clearly
        raise ImportError(
            "OffloadAdamW lives in the private module "
            "ramtorch.offload_optimizer; it may have moved in this RamTorch "
            f"version ({e})."
        ) from e

    import torch

    opts = []
    for dev, params in chunk_params_by_stage(
        chunks, chunks_per_stage, devices, exclude_ids=exclude_ids
    ):
        if not params:
            continue
        opts.append(OffloadAdamW(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            device=dev,
            bucket_mb=bucket_mb,
            window=window,
            state_dtype=state_dtype or torch.float32,
            stochastic_rounding=stochastic_rounding,
        ))
    if not opts:
        raise ValueError("no trainable parameters found in the chunk list")
    return MultiOptimizer(opts)

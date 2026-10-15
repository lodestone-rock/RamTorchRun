"""Temporary worker-local no-grad for nonconcurrent Krea preview inference."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager

import torch


@contextmanager
def inference_no_grad(pipe):
    """Run a pipeline's stage forwards without building preview autograd graphs.

    RamTorch's ``Pipeline.infer`` caller decorator does not propagate grad mode
    into its stage worker threads. Temporarily decorate each stage's existing
    bound forward instead, so ``no_grad`` is entered in the executing thread.
    This also works when a stage is already an offload wrapper.

    Use around the entire preview, with a separate context for the encoder if
    desired. Do not run training, other module calls, or overlapping contexts
    from other threads on the same stages while this context is active. Nested
    contexts in the same thread are supported. Synchronous ``infer`` joins its
    workers before returning/raising, so restoring forwards on exit is safe.

    Modules, parameters, buffers, training flags, and state-dict names are not
    replaced or changed. Caller grad mode is unchanged; retain the preview's
    ``@torch.no_grad()`` for sampling/VAE operations outside stage forwards.
    Both an existing instance forward override and an inherited class forward
    are restored exactly, including when inference or context setup raises.
    """
    with ExitStack() as restore:
        seen = set()
        for stage in pipe.stages:
            module = stage.module
            if id(module) in seen:
                continue
            seen.add(id(module))
            forward = module.forward
            had_override = "forward" in module.__dict__
            original_override = module.__dict__.get("forward")
            module.forward = torch.no_grad()(forward)
            if had_override:
                restore.callback(setattr, module, "forward", original_override)
            else:
                restore.callback(delattr, module, "forward")
        yield pipe

"""Opt-in, lazy transformer-block compilation for the ordinary Krea trainer.

``compile`` may be absent/false (eager), true, or an object with these defaults::

    {"dit": true, "encoder": true, "backend": "inductor",
     "mode": "default", "dynamic": true, "fullgraph": false}

Only resident ``parallelism: "pipeline"`` is supported. The backend may also
be ``aot_eager`` or ``eager`` for diagnosis/CPU checks. Only default mode is
supported; Inductor CUDA graphs are explicitly disabled for threaded Pipeline
execution. No arbitrary backend options or whole-model compilation are exposed.

Call after Pipeline placement and checkpoint flags, before the first forward.
We compile each block's *bound forward*, not its owning module/chunk: this
preserves parameter identity/state-dict paths and also covers callers that use
``layer.forward(...)`` directly (which would bypass ``nn.Module.compile``).
Embeddings, projections, heads, chunk orchestration, and VAE remain eager.
For threaded Pipeline workers, forward dispatch shares Dynamo's process lock
and backward lowering is forced into that protected forward compilation. There
is no CUDA synchronization and compiled backward execution remains unlocked.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps

import torch
from torch import nn

from krea2.model.chunks import DiTBlockChunk, DiTEmbedChunk, Qwen3VLEncoderChunk
from krea2.model.mmdit import SingleStreamBlock, TextFusionBlock


_DEFAULTS = dict(dit=True, encoder=True, backend="inductor", mode="default",
                 dynamic=True, fullgraph=False)


def validate_compile_config(cfg: Mapping) -> dict | None:
    """Validate before device setup/expensive models; never mutate the config."""
    raw = cfg.get("compile", False)
    if raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("compile must be a bool or an options object")
    unknown = raw.keys() - _DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown compile options: {sorted(unknown)}")
    options = {**_DEFAULTS, **raw}
    for key in ("dit", "encoder", "dynamic", "fullgraph"):
        if not isinstance(options[key], bool):
            raise ValueError(f"compile.{key} must be a bool")
    if options["backend"] not in ("inductor", "aot_eager", "eager"):
        raise ValueError("compile.backend must be 'inductor', 'aot_eager', or 'eager'")
    if options["mode"] != "default":
        raise ValueError("compile.mode must be 'default' (CUDA graphs are unsupported)")
    if not options["dit"] and not options["encoder"]:
        return None
    if cfg.get("parallelism", "offload") != "pipeline":
        raise ValueError("compile requires resident parallelism='pipeline'; weight streaming is unsupported")
    if not callable(getattr(torch, "compile", None)):
        raise RuntimeError("compile requires a PyTorch version with torch.compile")
    return options


def transformer_blocks(chunks: Sequence[nn.Module], target: str) -> list[tuple[str, nn.Module]]:
    """Select only executed transformer blocks, never their parent containers.

    Encoder chunks omit unused final decoder layers, final norm, vision tower
    and LM head. Select from those chunks rather than traversing all of Qwen.
    Names returned are canonical model paths, independent of chunk grouping.
    """
    selected = []
    if target == "dit":
        index = 0
        for chunk in chunks:
            if isinstance(chunk, DiTEmbedChunk):
                for group in ("layerwise_blocks", "refiner_blocks"):
                    for i, block in enumerate(getattr(chunk.txtfusion, group)):
                        if not isinstance(block, TextFusionBlock):
                            raise TypeError(f"unexpected text-fusion block: {type(block).__name__}")
                        selected.append((f"txtfusion.{group}.{i}", block))
            elif isinstance(chunk, DiTBlockChunk):
                for block in chunk.blocks:
                    if not isinstance(block, SingleStreamBlock):
                        raise TypeError(f"unexpected DiT block: {type(block).__name__}")
                    selected.append((f"blocks.{index}", block))
                    index += 1
    elif target == "encoder":
        for chunk in chunks:
            if not isinstance(chunk, Qwen3VLEncoderChunk):
                raise TypeError(f"unexpected encoder chunk: {type(chunk).__name__}")
            selected.extend((f"model.language_model.layers.{i}", layer)
                            for i, layer in zip(chunk.layer_indices, chunk.layers))
    else:
        raise ValueError(f"unknown compile target: {target!r}")
    if not selected:
        raise ValueError(f"no {target} transformer blocks found")
    if len({id(block) for _, block in selected}) != len(selected):
        raise ValueError(f"duplicate {target} transformer blocks")
    return selected


def _thread_safe_compiled_forward(forward, kwargs):
    # FX make_fx temporarily patches Module.__call__/__getattr__ and a process-
    # global tracing flag. Dynamo's own lock is acquired AFTER checking that
    # flag, so a second Pipeline worker can fail before it reaches that lock.
    # Share Dynamo's reentrant lock (not a second lock with inverted ordering).
    from torch._dynamo.convert_frame import compile_lock
    from torch._functorch import config as aot_config

    if not hasattr(aot_config, "force_non_lazy_backward_lowering"):
        raise RuntimeError("threaded block compilation requires PyTorch support for "
                           "force_non_lazy_backward_lowering")
    compiled = torch.compile(forward, **kwargs)

    @wraps(forward)
    def guarded(*args, **kw):
        with compile_lock:
            # Lower backward here too: otherwise the autograd engine can run
            # its first-use compiler outside this lock, racing another forward.
            # The option also makes backward lowering failures propagate rather
            # than silently deferring them to the unguarded backward path.
            with aot_config.patch(force_non_lazy_backward_lowering=True):
                return compiled(*args, **kw)

    # No CUDA synchronization: this only serializes Python tracing/dispatch;
    # previously enqueued kernels and already-compiled backwards can overlap.
    return guarded


def compile_transformer_blocks(
    chunks: Sequence[nn.Module], options: dict | None, *, target: str,
) -> tuple[str, ...]:
    """Install lazy compiled forwards in place; return selected block names.

    Options must come from ``validate_compile_config``. No forward, warmup,
    device operation, or parameter replacement occurs here. Compilation errors
    on first use are deliberately not swallowed or retried with eager execution.
    """
    if options is None or not options[target]:
        return ()
    selected = transformer_blocks(chunks, target)
    for name, block in selected:
        if hasattr(block, "_krea_compile_options"):
            raise ValueError(f"block already compiled: {name}")
    kwargs = dict(backend=options["backend"], dynamic=options["dynamic"],
                  fullgraph=options["fullgraph"])
    if options["backend"] == "inductor":
        # torch.compile rejects specifying mode AND options. Default mode is
        # implicit when supplying options; pin cudagraphs off even if a process
        # has changed the global Inductor default elsewhere.
        # Torch 2.10's mixed-order RMSNorm backward fusion explicitly forces
        # persistent kernels, BYPASSING persistent_reductions=False. At text
        # width 2560 it needs 180312 bytes of shared memory on sm120 (101376
        # available). Disable that fusion too; retain compiled tiled reductions
        # and native RMSNorm math rather than falling back to eager execution.
        kwargs["options"] = {"triton.cudagraphs": False,
                             "triton.persistent_reductions": False,
                             "triton.mix_order_reduction": False}
    else:
        kwargs["mode"] = options["mode"]
    for _, block in selected:
        block.forward = _thread_safe_compiled_forward(block.forward, kwargs)
        block._krea_compile_options = dict(options)
    names = tuple(name for name, _ in selected)
    print(f"  Compile {target}: {len(names)} transformer blocks, "
          f"backend={options['backend']} mode=default dynamic={options['dynamic']} "
          f"fullgraph={options['fullgraph']}; lazy, CUDA graphs disabled; "
          "thread-locked forward, eager backward lowering.")
    return names

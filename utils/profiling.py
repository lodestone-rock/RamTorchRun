"""profiling.py — Chrome-trace capture over a window of a sampling loop.

Generic (model-agnostic) helper for profiling a few iterations in the middle
of a long loop, so the trace stays small enough to open in Perfetto.

Why not the built-in RamTorch hooks: ``OffloadModel.step(profile_path=...)``
is the training path (inference goes through ``forward()``, which has no
profiling hook), and ``Pipeline.infer(profile_path=...)`` captures exactly one
``infer()`` call — but a diffusion step is two calls (cond + uncond), and we
want several consecutive steps in ONE timeline. So the profiler is driven from
the loop instead, which works identically for both execution modes.

For an ``OffloadModel`` we additionally splice in the H2D-loader and
D2H-writeback spans from its worker threads (kineto only records
``record_function`` annotations on the thread that entered the profiler, so
without this the streaming is invisible and only the compute shows up). This
reuses the same mechanism ``OffloadModel.step(profile_path=...)`` uses. Pass
several engines (one per streamed pipeline stage) and their tracks are
prefixed ``s0 ``, ``s1 ``, ... so each GPU's streaming reads separately.

Usage::

    tc = TraceCapture("trace.json", warmup=1, active=3, offload_models=[model])
    for i, t in enumerate(timesteps):
        with tc.iteration(i):
            ...                      # one diffusion step
        if tc.done:
            break
    tc.close()
"""
from __future__ import annotations

import contextlib
import os
import time
from typing import Optional

import torch
from torch.profiler import record_function


class TraceCapture:
    """Profile iterations ``[warmup, warmup + active)`` of a loop.

    ``warmup`` iterations are skipped so one-time costs (allocator growth,
    cuDNN autotuning, the first weight stream into a cold window) do not
    dominate the trace. ``done`` flips to True once the window has closed,
    letting the caller break out early instead of finishing the full run.
    """

    def __init__(
        self,
        path: str,
        *,
        warmup: int = 1,
        active: int = 3,
        offload_model=None,
        offload_models: Optional[list] = None,
        devices: Optional[list] = None,
    ):
        self.path = path
        self.warmup = max(0, warmup)
        self.active = max(1, active)
        self.offload_models = list(offload_models or [])
        if offload_model is not None:
            self.offload_models.append(offload_model)
        self.devices = devices or []
        self.done = False
        self._prof = None
        self._sync_wall_us = 0.0
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)

    # ── loop integration ───────────────────────────────────────────────────
    @contextlib.contextmanager
    def iteration(self, i: int):
        """Wrap one loop iteration; starts/stops the profiler at the edges."""
        if not self.done and i == self.warmup:
            self._start()
        if self._prof is None:
            yield
            return
        with record_function(f"diffusion_step_{i}"):
            yield
        if i + 1 >= self.warmup + self.active:
            self._stop()

    @contextlib.contextmanager
    def span(self, name: str):
        """Annotate a sub-region (e.g. "cond" / "uncond") inside an iteration."""
        if self._prof is None:
            yield
            return
        with record_function(name):
            yield

    # ── lifecycle ──────────────────────────────────────────────────────────
    def _start(self):
        from torch.profiler import ProfilerActivity, profile as _profile

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        self._prof = _profile(activities=activities, record_shapes=False,
                              with_stack=False)
        self._prof.__enter__()
        # The loader/writeback threads log spans against time.monotonic_ns;
        # this marker pairs a kineto timestamp with a monotonic reading so the
        # two clocks can be aligned when splicing them in.
        for m in self.offload_models:
            m._span_log = []
        with record_function("offload_clock_sync"):
            self._sync_wall_us = time.monotonic_ns() / 1e3
        print(f"[profile] capturing {self.active} iteration(s) -> {self.path}")

    def _stop(self):
        if self._prof is None:
            return
        # Drain device work BEFORE stopping so trailing kernels/copies land
        # in the trace.
        for d in self.devices or ([torch.device("cuda")] if torch.cuda.is_available() else []):
            if torch.device(d).type == "cuda":
                torch.cuda.synchronize(d)
        self._prof.__exit__(None, None, None)
        self._prof.export_chrome_trace(self.path)
        # One injection pass for all engines: the injector allocates trace
        # thread ids per distinct track NAME, so per-stage names must be made
        # unique first or several stages would share one track.
        all_spans, injector = [], None
        multi = len(self.offload_models) > 1
        for k, m in enumerate(self.offload_models):
            spans, m._span_log = m._span_log, None
            if not spans:
                continue
            injector = type(m)._inject_thread_spans
            all_spans += [
                ((f"s{k} {track}" if multi else track), name, t_us, dur_us)
                for track, name, t_us, dur_us in spans
            ]
        if all_spans:
            injector(self.path, all_spans, self._sync_wall_us)
        self._prof = None
        self.done = True
        size_mb = os.path.getsize(self.path) / 1e6
        print(f"[profile] wrote {self.path} ({size_mb:.1f} MB)")

    def close(self):
        """Stop early if the loop ended before the window closed."""
        self._stop()

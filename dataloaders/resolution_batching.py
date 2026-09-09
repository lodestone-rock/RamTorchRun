"""Configuration and batch contracts for opt-in step-level resolution sampling.

Trainer example::

    "resolution_batching": {
        "effective_batch_size": 16,
        "resolutions": {
            "256": {"probability": 0.95, "microbatch_size": 16},
            "1024": {"probability": 0.05, "microbatch_size": 1}
        },
        "steps_per_epoch": 1000
    }

Probabilities count optimizer steps, not images assigned to pools or wall time.
The effective batch is fixed; accumulation is E / microbatch_size. The block
supersedes legacy batch/accumulation/base-resolution settings only when enabled.
Resume data plans with parquet_dataloader.data_epoch and .offset (next step in
that epoch). This does not restore optimizer or augmentation RNG state.
"""

import math
from collections.abc import Mapping
from numbers import Real


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def normalize_resolution_batching(config):
    """Validate without mutating input; normalize JSON resolution keys to ints."""
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("resolution_batching must be an object")
    unknown = set(config) - {"effective_batch_size", "resolutions", "steps_per_epoch"}
    if unknown:
        raise ValueError(f"Unknown resolution_batching settings: {sorted(unknown)}")
    effective = _positive_int(config.get("effective_batch_size"), "effective_batch_size")
    resolutions = config.get("resolutions")
    if not isinstance(resolutions, Mapping) or not resolutions:
        raise ValueError("resolutions must be a nonempty object")
    normalized = {}
    for key, settings in resolutions.items():
        if isinstance(key, str) and key.isascii() and key.isdecimal():
            resolution = int(key)
        else:
            resolution = key
        resolution = _positive_int(resolution, "resolution")
        if resolution in normalized:
            raise ValueError(f"Duplicate normalized resolution: {resolution}")
        if not isinstance(settings, Mapping) or set(settings) != {"probability", "microbatch_size"}:
            raise ValueError(f"Resolution {resolution} needs probability and microbatch_size only")
        microbatch = _positive_int(settings["microbatch_size"], "microbatch_size")
        if effective % microbatch:
            raise ValueError(f"effective_batch_size {effective} must be divisible by microbatch_size {microbatch}")
        probability = settings["probability"]
        if (isinstance(probability, bool) or not isinstance(probability, Real)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("probability must be finite and in [0, 1]")
        normalized[resolution] = {"probability": float(probability), "microbatch_size": microbatch}
    total = math.fsum(item["probability"] for item in normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Resolution probabilities must sum to 1, got {total}")
    steps = config.get("steps_per_epoch")
    if steps is not None:
        steps = _positive_int(steps, "steps_per_epoch")
    return {"effective_batch_size": effective, "resolutions": normalized, "steps_per_epoch": steps}


def resolution_batching_kwargs(cfg):
    """Build the opt-in dataset kwargs from a trainer's configuration."""
    config = normalize_resolution_batching(cfg.get("resolution_batching"))
    if config is None:
        return {}
    parquet = cfg.get("parquet_dataloader") or {}
    if cfg.get("parquet_dataloader") is not None:
        validate_resolution_geometry(config, parquet.get("ratio_cutoff", 2.0),
                                     parquet.get("resolution_step", 64))
    epoch = parquet.get("data_epoch", 0)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("parquet_dataloader.data_epoch must be a nonnegative integer")
    return {"resolution_batching": config, "data_epoch": epoch}


def validate_resolution_geometry(config, ratio_cutoff, step):
    """Guard the existing bucket generator against zero sides and invalid loops."""
    _positive_int(step, "resolution_step")
    if (isinstance(ratio_cutoff, bool) or not isinstance(ratio_cutoff, Real)
            or not math.isfinite(ratio_cutoff) or ratio_cutoff <= 1):
        raise ValueError("ratio_cutoff must be finite and greater than 1")
    # Simulate only the aspect progression; leave the legacy generator untouched.
    for resolution in config["resolutions"]:
        x = y = resolution
        while y / x <= ratio_cutoff:
            y += step
            x = int((resolution ** 2 / y) // step * step)
            if x <= 0:
                raise ValueError(f"Resolution {resolution}, step {step}, ratio_cutoff {ratio_cutoff} "
                                 "generate a zero-sized bucket; use a smaller resolution_step")


def validate_resolution_alignment(config, parquet_config, alignment):
    """Reject bucket geometry incompatible with the model before loading weights.

    Square bases and the aspect bucket step must both align. The image models
    use VAE compression * patch, or pixel patch_size for VAE-free Radiance.
    """
    if config is None:
        return
    config = normalize_resolution_batching(config)
    _positive_int(alignment, "model alignment")
    step = parquet_config.get("resolution_step", 64)
    _positive_int(step, "resolution_step")
    if step % alignment:
        raise ValueError(f"resolution_step {step} must be divisible by model alignment {alignment}")
    for resolution in config["resolutions"]:
        if resolution % alignment:
            raise ValueError(f"Resolution {resolution} must be divisible by model alignment {alignment}")


def get_step_batching(batch, cfg_resolution_batching, legacy_n_mb):
    """Return (accumulation, microbatch size, base resolution) for actual data.

    The last extras dictionary also carries tags; never infer a resolution
    family from a rounded rectangular bucket's pixel area.
    """
    batch_size = batch[0].shape[0]
    if cfg_resolution_batching is None:
        return legacy_n_mb, batch_size // legacy_n_mb, None
    config = normalize_resolution_batching(cfg_resolution_batching)
    extras = batch[-1] if isinstance(batch[-1], dict) else {}
    plan = extras.get("step_plan")
    if not isinstance(plan, dict):
        raise ValueError("resolution_batching requires step_plan batch metadata")
    resolution = plan.get("resolution")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution not in config["resolutions"]:
        raise ValueError("Batch resolution is missing or not configured")
    settings = config["resolutions"][resolution]
    if settings["probability"] <= 0:
        raise ValueError("Batch selected a zero-probability resolution")
    effective = config["effective_batch_size"]
    microbatch = settings["microbatch_size"]
    accumulation = effective // microbatch
    expected = {"effective_batch_size": effective, "microbatch_size": microbatch,
                "n_microbatches": accumulation}
    if batch_size != effective:
        raise ValueError(f"Planned effective batch {effective}, received {batch_size}")
    for name, value in expected.items():
        if type(plan.get(name)) is not int or plan[name] != value:
            raise ValueError(f"Inconsistent step_plan {name}: expected {value}")
    bucket = plan.get("bucket")
    if (not isinstance(bucket, (tuple, list)) or len(bucket) != 2
            or any(type(side) is not int or side <= 0 for side in bucket)
            or tuple(batch[0].shape[-2:]) != (bucket[1], bucket[0])):
        raise ValueError("Batch image shape does not match step_plan bucket (width, height)")
    if len(batch[1]) != effective or len(batch[3]) != effective:
        raise ValueError("Captions/loss weights do not match effective batch")
    return accumulation, microbatch, resolution


def resolution_batching_summary(config):
    config = normalize_resolution_batching(config)
    if config is None:
        return "Resolution batching: disabled (legacy settings)"
    effective = config["effective_batch_size"]
    entries = [f"{res}px: {item['probability']:.1%} steps, "
               f"{item['microbatch_size']} x {effective // item['microbatch_size']} accumulation"
               for res, item in config["resolutions"].items()]
    return (f"Resolution batching: effective batch={effective}; " + "; ".join(entries)
            + ". Overrides legacy batch_size, n_microbatches/grad_accum and base-resolution weights.")

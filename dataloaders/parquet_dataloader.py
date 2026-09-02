"""Parquet-based dataloader for multi-source image-text datasets.

Supports:
- Loading from Hive-partitioned parquet directories (source=xxx/part-*.parquet)
- Per-source subsampling with optional repeat + warning
- Multi-caption sampling with configurable probabilities
- Tag-based vs caption-based flag per caption column
- Bucketing, round-robin multi-GPU sharding, reference images
- Same return signature as TextImageDataset

Example config snippet:
{
    "parquet_sources": {
        "e621":       {"path": "/data/parquet/source=e621",       "n_samples": 1000000},
        "deviantart": {"path": "/data/parquet/source=deviantart", "n_samples": 500000}
    },
    "caption_columns": {
        "tags":                    {"weight": 0.2, "is_tag_based": true},
        "midjourney_style_summary": {"weight": 0.3, "is_tag_based": false},
        "brief_summary":           {"weight": 0.5, "is_tag_based": false}
    },
    "filename_column":    "url",
    "width_column":       "image_width",
    "height_column":      "image_height",
    "loss_weight_column": null,          // or e.g. "sampling_probability"
    "image_folder_path":  "",            // base dir for local files; ignored for URLs
    "base_res":           [512, 1024],   // multiple base resolutions
    "base_res_weights":   [0.25, 0.75], // optional: weighted sampling; omit for uniform
    ...
}
"""

import os
import math
import random
import logging
import warnings
from io import BytesIO

import concurrent.futures as _cf

# Must be set before cv2 is imported so the OpenEXR codec is enabled.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

try:
    import tifffile as _tifffile
    _TIFFFILE_AVAILABLE = True
except ImportError:
    _tifffile = None
    _TIFFFILE_AVAILABLE = False

from .bucketing_logic import (
    _bucket_generator,
    _normalize_width_height,
    _closest_bucket,
)
from . import color_profile_handling

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HDR / high-bit-depth image helpers
# ---------------------------------------------------------------------------

# Magic-byte signatures for formats that need special handling.
# Checked against the first 16 bytes of the file/response body.
_MAGIC_EXR   = b"\x76\x2f\x31\x01"          # OpenEXR
_MAGIC_HDR1  = b"#?RADIANCE"                  # Radiance RGBE (.hdr)
_MAGIC_HDR2  = b"#?RGBE"                      # alternate Radiance header
_MAGIC_TIFF_LE = b"II*\x00"                   # TIFF little-endian
_MAGIC_TIFF_BE = b"MM\x00*"                   # TIFF big-endian
_MAGIC_PNG   = b"\x89PNG"                     # PNG (any bit depth)
_MAGIC_JXL_BARE = b"\xff\x0a"                 # JXL bare codestream
_MAGIC_JXL_BMFF = b"\x00\x00\x00\x0c\x4a\x58\x4c\x20"  # JXL ISO BMFF box


def _detect_hdr_format(header: bytes) -> str:
    """Sniff the first bytes of image data and return a format tag.

    Returns
    -------
    ``"exr"``      – OpenEXR
    ``"hdr"``      – Radiance RGBE (.hdr)
    ``"tiff"``     – TIFF (any bit depth / sample type)
    ``"png"``      – PNG (may be 8- or 16-bit; caller must check)
    ``"jxl"``      – JPEG XL
    ``"standard"`` – everything else (JPEG, WEBP, …)
    """
    if header[:4] == _MAGIC_EXR:
        return "exr"
    if header[:10] == _MAGIC_HDR1 or header[:6] == _MAGIC_HDR2:
        return "hdr"
    if header[:4] in (_MAGIC_TIFF_LE, _MAGIC_TIFF_BE):
        return "tiff"
    if header[:4] == _MAGIC_PNG:
        return "png"
    if header[:2] == _MAGIC_JXL_BARE or header[:8] == _MAGIC_JXL_BMFF:
        return "jxl"
    return "standard"


def _load_as_float32(data: bytes, fmt: str) -> np.ndarray:
    """Decode *data* into a float32 H×W×3 array in linear light.

    Supported *fmt* values: ``"exr"``, ``"hdr"``, ``"tiff"``.
    For 16-bit PNG pass ``"png"`` — the caller is responsible for
    confirming the bit depth before calling this function.

    Raises ``ValueError`` if the format is unsupported or decoding fails.
    """
    if fmt in ("exr", "hdr"):
        # cv2 with IMREAD_ANYDEPTH returns float32 for EXR/HDR.
        buf = np.frombuffer(data, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError(f"cv2 failed to decode {fmt} image")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr.astype(np.float32)

    if fmt == "tiff":
        if not _TIFFFILE_AVAILABLE:
            raise RuntimeError("tifffile is required to load TIFF images; install it with: pip install tifffile")
        arr = _tifffile.imread(BytesIO(data))
        # Ensure H×W×3 float32
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype == np.uint8:
            # 8-bit TIFF — not HDR, but handle gracefully
            return arr.astype(np.float32) / 255.0
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 65535.0
        if arr.dtype == np.uint32:
            return arr.astype(np.float32) / 4294967295.0
        # float16 / float32 / float64 — assume linear light
        return arr.astype(np.float32)

    if fmt == "png":
        # 16-bit PNG: cv2 ANYDEPTH returns uint16.
        buf = np.frombuffer(data, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("cv2 failed to decode 16-bit PNG")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 65535.0
        # Fell back to 8-bit — shouldn't reach here but handle it
        return arr.astype(np.float32) / 255.0

    raise ValueError(f"_load_as_float32: unsupported format '{fmt}'")


def _reinhard_tonemap_to_pil(arr: np.ndarray) -> Image.Image:
    """Tonemap a linear float32 H×W×3 array to an 8-bit sRGB PIL Image.

    Uses the Reinhard global operator followed by the IEC 61966-2-1 sRGB
    transfer function (proper gamma, not a simple power-law approximation).
    """
    # Reinhard global: maps [0, ∞) → [0, 1)
    arr = arr / (1.0 + arr)
    # sRGB transfer function (IEC 61966-2-1)
    srgb = np.where(
        arr <= 0.0031308,
        arr * 12.92,
        1.055 * np.power(np.clip(arr, 0.0, None), 1.0 / 2.4) - 0.055,
    )
    srgb = np.clip(srgb, 0.0, 1.0)
    return Image.fromarray((srgb * 255.0).round().astype(np.uint8))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_parquet_files(path: str) -> list[str]:
    """Walk *path* and return all .parquet file paths, sorted."""
    if os.path.isfile(path) and path.endswith(".parquet"):
        return [path]
    files = []
    for root, _dirs, fnames in os.walk(path):
        for f in sorted(fnames):
            if f.endswith(".parquet"):
                files.append(os.path.join(root, f))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    return files


def _read_parquet_columns(filepath: str, columns: list[str]) -> pa.Table:
    """Read only *columns* from a single parquet file."""
    # Use the file-level schema to skip missing columns gracefully
    schema = pq.read_schema(filepath)
    existing = [c for c in columns if c in schema.names]
    # ParquetFile.read() instead of pq.read_table(): the dataset API refuses
    # to merge row groups whose encodings disagree with the file-level schema
    # (e.g. large_string footer + dictionary-encoded row groups), while the
    # per-file reader decodes them transparently.
    return pq.ParquetFile(filepath).read(columns=existing)


def _cast_strings_to_large(table: pa.Table) -> pa.Table:
    """Cast all string/binary columns to large_string/large_binary.

    PyArrow's string type uses 32-bit offsets (max ~2 GB per column chunk).
    When concatenating many files the offset can overflow; large_string uses
    64-bit offsets and avoids the error entirely.
    """
    new_fields = []
    new_cols = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        # Dictionary-encoded strings (some parquet writers use them) must be
        # decoded first, otherwise concat_tables fails against large_string.
        if pa.types.is_dictionary(field.type):
            col = pa.compute.dictionary_decode(col)
            field = field.with_type(col.type)
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            col = col.cast(pa.large_utf8())
            field = field.with_type(pa.large_utf8())
        elif pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
            col = col.cast(pa.large_binary())
            field = field.with_type(pa.large_binary())
        new_fields.append(field)
        new_cols.append(col)
    return pa.table({f.name: c for f, c in zip(new_fields, new_cols)})


def _coerce_text_columns(table: pa.Table, columns: list[str]) -> pa.Table:
    """Force *columns* to large_string so sources with different builds concat.

    `_cast_strings_to_large` is driven by the type it finds, so it cannot help
    when a source stores a text column as something else entirely. A build that
    has no data for a caption column may type it from the nulls it wrote — the
    v2 booru corpus writes `10_word_summary` as an all-null int32, which
    `concat_tables` refuses to merge against another source's large_string.

    A non-string text column carries no caption by definition, so it becomes
    all-null. Missing columns are added as null too, which keeps the projection
    rectangular when only some sources have a column at all.
    """
    for name in columns:
        idx = table.schema.get_field_index(name)
        if idx >= 0 and pa.types.is_large_string(table.schema.field(idx).type):
            continue
        if idx >= 0 and pa.types.is_string(table.schema.field(idx).type):
            col = table.column(idx).cast(pa.large_utf8())
        else:
            col = pa.nulls(len(table), type=pa.large_utf8())
        field = pa.field(name, pa.large_utf8())
        table = (table.set_column(idx, field, col) if idx >= 0
                 else table.append_column(field, col))
    return table


def _load_parquet_source(
    path: str,
    n_samples: int | None,
    seed: int,
    columns: list[str],
    num_threads: int = 8,
) -> pa.Table:
    """Load parquet files under *path* in parallel, project *columns*, subsample.

    Returns a PyArrow Table — no pandas involved.
    Uses its own seeded RNG so callers can run this in parallel safely.
    """
    rng = random.Random(seed)
    parquet_files = _collect_parquet_files(path)

    # Parallel reads — each worker reads one file
    with _cf.ThreadPoolExecutor(max_workers=min(num_threads, len(parquet_files))) as ex:
        futures = {ex.submit(_read_parquet_columns, f, columns): f for f in parquet_files}
        tables = []
        for fut in _cf.as_completed(futures):
            tables.append(fut.result())

    # Cast strings to large_string before concat to avoid 32-bit offset overflow
    tables = [_cast_strings_to_large(t) for t in tables]

    # Concatenate with zero-copy where possible
    table = pa.concat_tables(tables, promote_options="default")
    del tables
    total = len(table)

    if n_samples is not None:
        if n_samples > total:
            warnings.warn(
                f"Requested {n_samples} samples from '{path}' but only {total} rows "
                f"are available. Repeating rows to reach the target.",
                stacklevel=3,
            )
            repeats = math.ceil(n_samples / total)
            idx = list(range(total)) * repeats
            rng.shuffle(idx)
            idx = idx[:n_samples]
        else:
            idx = rng.sample(range(total), n_samples)
        table = table.take(pa.array(idx, type=pa.int64()))

    return table


def _build_standardized_buckets(base_resolution: list[int], ratio_cutoff: float, step: int) -> dict:
    """Return {res: {(norm_w, norm_h): (w, h), ...}, ...}."""
    standardized_buckets = {}
    for res in base_resolution:
        buckets = _bucket_generator(res, ratio_cutoff, step)
        this_std = {}
        for bucket in buckets:
            b_st = _normalize_width_height(*bucket)
            this_std[b_st] = bucket
        standardized_buckets[res] = this_std
    return standardized_buckets


def _assign_bucket(
    image_width: int,
    image_height: int,
    standardized_buckets: dict,
    ratio_cutoff: float,
    rng: random.Random,
    weights: list[float] | None = None,
) -> tuple | None:
    """Return a randomly chosen bucket (w, h) for the given image dimensions, or None if filtered.

    Parameters
    ----------
    weights :
        Sampling probabilities for each entry in ``standardized_buckets``
        (must match its length).  ``None`` → uniform selection (original
        behaviour, preserved for backward compatibility).
    """
    if image_width <= 0 or image_height <= 0:
        return None
    aspect_ratio = image_width / image_height
    if not (1.0 / ratio_cutoff < aspect_ratio < ratio_cutoff):
        return None
    w, h = _normalize_width_height(image_width, image_height)
    bucket_list = list(standardized_buckets.values())
    chosen_std = rng.choices(bucket_list, weights=weights, k=1)[0]
    return _closest_bucket(w, h, chosen_std)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class ParquetTextImageDataset(Dataset):
    """Multi-source parquet dataset with bucketing and multi-GPU sharding.

    Parameters
    ----------
    batch_size : int
        Global batch size (across all GPUs).
    parquet_sources : dict
        ``{source_name: {"path": str, "n_samples": int | None}}``
        ``n_samples`` is optional; omit or set to ``null`` to use all rows.
    caption_columns : dict
        ``{col_name: {"weight": float, "is_tag_based": bool}}``
        Weights are normalised internally.
    filename_column : str
        Column that holds the image filename or URL.
    width_column : str
        Column for image width.
    height_column : str
        Column for image height.
    loss_weight_column : str | None
        Column for per-sample loss weight. If ``None``, defaults to 1.
    image_folder_path : str
        Base directory prepended to local filenames. Ignored for URLs.
    base_res : list[int]
        Base resolutions for bucket generation.
    base_res_weights : list[float] | None
        Optional sampling weights for each entry in ``base_res``.  When
        provided, buckets from ``base_res[i]`` are chosen with probability
        proportional to ``base_res_weights[i]``; otherwise every resolution
        is equally likely (original behaviour).  Raw positive values are
        accepted — they are normalised internally, so ``[1, 3]`` and
        ``[0.25, 0.75]`` are equivalent.
    ratio_cutoff : float
        Maximum aspect ratio (images outside ``[1/r, r]`` are dropped).
    resolution_step : int
        Step size for bucket generation.
    shuffle_tags : bool
        Shuffle comma-separated tags when ``is_tag_based`` is True.
    tag_drop_percentage : float
        Fraction of tags that may be randomly dropped (0 = keep all).
    uncond_percentage : float
        Probability of replacing caption with empty string (CFG dropout).
    seed : int
        Master random seed.
    rank : int
        This GPU's rank (0-indexed).
    num_gpus : int
        Total number of GPUs.
    timeout : int
        HTTP request timeout in seconds.
    thread_per_worker : int
        Thread pool size for concurrent image loading.
    dummy_image : bool
        If True, skip actual image loading (for debugging).
    offset : int
        Skip the first *offset* batches.
    num_reference_images : int | None
        If set, load this many reference images per sample from the
        ``reference_images`` column (list of filenames/URLs).
    tag_column : str | None
        Column holding a comma-separated tag list. When set, ``__getitem__``
        appends ONE extra element to its return tuple: a dict with
        ``tag_ids`` ``[B, max_tags]`` int64 and ``tag_mask`` ``[B, max_tags]``
        bool. The ids are looked up in ``tag_vocab_path`` and are **decoupled
        from the caption** — a row whose caption came from a natural-language
        column still reports its full tag set, which is what lets a model learn
        to use both channels at once. A null/empty tag column gives an
        all-False mask. No dropout is applied here; that belongs to the trainer.
    tag_vocab_path : str | None
        Parquet written by ``utils/tag_vocab.py``. Required with ``tag_column``.
    max_tags : int
        Width of the tag block. Overflowing rows are randomly subsampled rather
        than truncated, since the corpus is mostly alphabetically sorted.
    raw_hdr : bool
        If ``False`` (default), HDR / high-bit-depth images are tonemapped
        (Reinhard global + sRGB gamma) and returned as normalised ``[-1, 1]``
        float32 tensors — identical to the standard 8-bit path.
        If ``True``, HDR images are returned as raw linear float32 tensors
        with shape ``[C, H, W]`` and **no** normalisation applied; the
        ``image_transforms`` pipeline is skipped for those samples.
    """

    def __init__(
        self,
        batch_size: int,
        parquet_sources: dict,
        caption_columns: dict,
        filename_column: str = "url",
        width_column: str = "image_width",
        height_column: str = "image_height",
        loss_weight_column: str | None = None,
        image_folder_path: str = "",
        base_res: list[int] = None,
        base_res_weights: list[float] | None = None,
        ratio_cutoff: float = 2.0,
        resolution_step: int = 64,
        shuffle_tags: bool = True,
        tag_drop_percentage: float = 0.1,
        uncond_percentage: float = 0.05,
        seed: int = 0,
        rank: int = 0,
        num_gpus: int = 1,
        timeout: int = 10,
        thread_per_worker: int = 100,
        dummy_image: bool = False,
        offset: int = 0,
        num_reference_images: int | None = None,
        raw_hdr: bool = False,
        tokenizer=None,
        max_text_len: int = 128,
        tag_column: str | None = None,
        tag_vocab_path: str | None = None,
        max_tags: int = 128,
    ):
        if base_res is None:
            base_res = [1024]

        # Validate and normalise base_res_weights
        if base_res_weights is not None:
            if len(base_res_weights) != len(base_res):
                raise ValueError(
                    f"base_res_weights length ({len(base_res_weights)}) must match "
                    f"base_res length ({len(base_res)})"
                )
            if any(w < 0 for w in base_res_weights):
                raise ValueError("All base_res_weights must be non-negative")
            total_res_w = sum(base_res_weights)
            if total_res_w <= 0:
                raise ValueError("base_res_weights must sum to a positive value")
            base_res_weights = [w / total_res_w for w in base_res_weights]

        assert batch_size % num_gpus == 0, "batch_size must be divisible by num_gpus"

        self.batch_size = batch_size
        self.rank_batch_size = batch_size // num_gpus
        self.parquet_sources = parquet_sources
        self.filename_column = filename_column
        self.width_column = width_column
        self.height_column = height_column
        self.loss_weight_column = loss_weight_column
        self.image_folder_path = image_folder_path
        self.base_res = base_res
        self.base_res_weights = base_res_weights
        self.ratio_cutoff = ratio_cutoff
        self.resolution_step = resolution_step
        self.shuffle_tags = shuffle_tags
        self.tag_drop_percentage = tag_drop_percentage
        self.uncond_percentage = uncond_percentage
        self.rank = rank
        self.num_gpus = num_gpus
        self.timeout = timeout
        self.thread_per_worker = thread_per_worker
        self.dummy_image = dummy_image
        self.num_reference_images = num_reference_images
        self.offset = offset
        self.raw_hdr = raw_hdr
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # Tag ids are DECOUPLED from the caption: they come from `tag_column`
        # whichever caption column was sampled, so a natural-language caption
        # still carries the row's full tag set. Rows whose tag column is null
        # (non-booru sources) yield an all-False mask, which the model treats
        # as an exact no-op. No dropout is applied here — the trainers own that
        # policy, next to their existing uncond handling.
        self.tag_column = tag_column
        self.max_tags = max_tags
        self.tag_matcher = None
        if tag_column is not None:
            if not tag_vocab_path:
                raise ValueError("tag_column requires tag_vocab_path")
            from utils.tag_vocab import TagMatcher, TagVocab

            # build_trie=False: the training tag column is already a clean
            # comma list, so only the exact-segment pass is needed. The trie is
            # ~300 MB and would be copied into every dataloader worker.
            self.tag_vocab = TagVocab.load(tag_vocab_path)
            self.tag_matcher = TagMatcher(self.tag_vocab, build_trie=False)

        # Normalise caption column weights
        self.caption_columns = caption_columns
        total_w = sum(v["weight"] for v in caption_columns.values())
        self._caption_col_names: list[str] = []
        self._caption_col_weights: list[float] = []
        self._caption_col_is_tag: list[bool] = []
        for col, cfg in caption_columns.items():
            self._caption_col_names.append(col)
            self._caption_col_weights.append(cfg["weight"] / total_w)
            self._caption_col_is_tag.append(bool(cfg.get("is_tag_based", False)))

        self._rng = random.Random(seed)

        self.image_transforms = v2.Compose(
            [v2.ToTensor(), v2.Normalize(mean=[0.5], std=[0.5])]
        )

        if _REQUESTS_AVAILABLE:
            import requests as _req
            self.session = _req.Session()
        else:
            self.session = None

        self.batches = self._load_batches()
        self._round_robin()
        self.batches = self.batches[offset:]

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _load_batches(self) -> list:
        rng = self._rng

        # Columns we actually need — avoids pulling heavy unused columns into RAM
        needed_cols = list({
            self.filename_column,
            self.width_column,
            self.height_column,
            *self._caption_col_names,
            *([] if not self.loss_weight_column else [self.loss_weight_column]),
            *([] if self.num_reference_images is None else ["reference_images"]),
            *([] if self.tag_column is None else [self.tag_column]),
        })

        # ---- 1. Load & subsample each source in parallel ----
        n_io_threads = min(8, psutil.cpu_count(logical=False) or 4)

        text_cols = list({
            *self._caption_col_names,
            *([] if self.tag_column is None else [self.tag_column]),
        })

        def _load_source(item):
            source_name, src_cfg = item
            path = src_cfg["path"]
            n_samples = src_cfg.get("n_samples", None)
            # Derive a deterministic but unique seed per source so parallel
            # calls don't share RNG state
            source_seed = rng.randint(0, 2**31)
            table = _load_parquet_source(path, n_samples, source_seed, needed_cols, n_io_threads)
            table = _coerce_text_columns(table, text_cols)
            print(f"  [{source_name}] loaded {len(table):,} rows")
            return table

        with _cf.ThreadPoolExecutor(max_workers=len(self.parquet_sources)) as ex:
            tables = list(ex.map(_load_source, self.parquet_sources.items()))

        combined = pa.concat_tables(tables, promote_options="default")
        del tables
        total_rows = len(combined)
        print(f"Total rows after subsampling: {total_rows:,}")

        # ---- 2. Build bucket lookup ----
        standardized_buckets = _build_standardized_buckets(
            self.base_res, self.ratio_cutoff, self.resolution_step
        )
        std_bucket_list = list(standardized_buckets.values())  # for rng.choice

        # ---- 3. Extract columns as Python lists once — avoids per-row PyArrow overhead ----
        col_names = combined.schema.names

        def _col(name):
            if name in col_names:
                return combined.column(name).to_pylist()
            return None

        filenames    = _col(self.filename_column)
        widths       = _col(self.width_column)
        heights      = _col(self.height_column)
        loss_weights = _col(self.loss_weight_column) if self.loss_weight_column else None
        ref_images   = _col("reference_images") if self.num_reference_images is not None else None
        tag_values   = _col(self.tag_column) if self.tag_column else None
        caption_cols = {c: _col(c) for c in self._caption_col_names}

        # Free the Arrow table — all data is now in plain Python lists
        del combined

        # ---- 4. Assign buckets and build sample dicts ----
        buckets: dict[tuple, list] = {}
        skipped = 0
        skipped_no_caption = 0

        for i in range(total_rows):
            try:
                w = int(widths[i])
                h = int(heights[i])
            except (TypeError, ValueError):
                skipped += 1
                continue

            bucket = _assign_bucket(w, h, standardized_buckets, self.ratio_cutoff, rng,
                                     weights=self.base_res_weights)
            if bucket is None:
                skipped += 1
                continue

            # Find which caption columns have a non-empty value for this row
            available_indices = [
                idx for idx, col_name in enumerate(self._caption_col_names)
                if caption_cols[col_name] is not None
                and caption_cols[col_name][i] is not None
                and str(caption_cols[col_name][i]).strip() != ""
            ]

            # Drop the row entirely if no caption column has a value
            if not available_indices:
                skipped_no_caption += 1
                continue

            # Sample only from available columns, re-normalising weights on the fly
            available_weights = [self._caption_col_weights[idx] for idx in available_indices]
            col_idx   = rng.choices(available_indices, weights=available_weights, k=1)[0]
            col_name  = self._caption_col_names[col_idx]
            is_tag    = self._caption_col_is_tag[col_idx]
            caption   = str(caption_cols[col_name][i]).strip()

            lw = 1.0
            if loss_weights is not None:
                try:
                    lw = float(loss_weights[i])
                except (TypeError, ValueError):
                    lw = 1.0

            filename = str(filenames[i])
            sample = {
                "filename":       filename,
                "caption_or_tags": caption,
                "bucket":         bucket,
                "is_tag_based":   is_tag,
                "is_url_based":   self._is_url(filename),
                "loss_weight":    lw,
                "reference_images": (ref_images[i] if ref_images and isinstance(ref_images[i], list) else []),
                # Raw string, resolved to ids per batch in the worker: keeping
                # 1.16M x ~44 int arrays resident would cost ~200 MB per rank
                # for a lookup that costs microseconds.
                "tags": (tag_values[i] if tag_values and tag_values[i] else ""),
            }

            if bucket in buckets:
                buckets[bucket].append(sample)
            else:
                buckets[bucket] = [sample]

        if skipped:
            log.warning(f"Skipped {skipped:,} rows (bad dimensions or extreme aspect ratio)")
        if skipped_no_caption:
            log.warning(f"Skipped {skipped_no_caption:,} rows (all caption columns were None/empty)")

        # ---- 5. Shuffle within each bucket ----
        for key in buckets:
            rng.shuffle(buckets[key])

        post_count = sum(len(v) for v in buckets.values())
        print(f"There are {post_count:,} text-image pairs across {len(buckets)} buckets.")

        # ---- 6. Pack into full batches ----
        # A batch must be exactly batch_size: every sample in it shares one
        # resolution, and `_round_robin` slices it per rank. The tail of a
        # bucket that is not a multiple of batch_size is therefore ECHOED from
        # elsewhere in the same bucket rather than dropped — dropping silently
        # discarded up to batch_size-1 samples *per bucket*, which for a small
        # or heavily-bucketed set could be most of the data, and if it took
        # every bucket to zero the only symptom was an IndexError from an empty
        # dataset. This matches what __getitem__ already does for a batch whose
        # images failed to load.
        batches = []
        n_echoed = 0
        for b in buckets.values():
            for i in range(0, len(b), self.batch_size):
                samples = b[i:i + self.batch_size]
                short = self.batch_size - len(samples)
                if short:
                    # Draw from the whole bucket, not just the tail, so the few
                    # leftover samples are not over-weighted.
                    samples = samples + [b[rng.randrange(len(b))]
                                         for _ in range(short)]
                    n_echoed += short
                batches.append(samples)

        if not batches:
            raise RuntimeError(
                f"no batches: {post_count:,} sample(s) across {len(buckets)} "
                f"bucket(s) and batch_size={self.batch_size}. Every bucket was "
                f"empty — check the caption/dimension columns and the aspect "
                f"ratio cutoff."
            )
        if n_echoed:
            log.info(
                f"Echoed {n_echoed:,} sample(s) to fill the last batch of "
                f"{len(buckets)} bucket(s) (was: dropped)."
            )
        print(f"We got {len(batches):,} batches.")
        rng.shuffle(batches)
        return batches

    def _round_robin(self):
        """Slice batches for this rank (each batch → rank's slice)."""
        subset = []
        for batch in self.batches:
            subset.append(
                batch[
                    self.rank * self.rank_batch_size:
                    self.rank * self.rank_batch_size + self.rank_batch_size
                ]
            )
        self.batches = subset

    def resample(self):
        """Draw a fresh subsample from the parquet sources and rebuild batches.

        Call this at the end of each epoch so the per-source subsampling picks
        a different random subset next time.  The internal RNG advances with
        every call, so successive epochs are always distinct.

        Example::

            for epoch in range(num_epochs):
                for batch in dataloader:
                    ...
                dataset.resample()   # refresh before next epoch
        """
        print(f"[ParquetTextImageDataset] resampling (rank {self.rank})...")
        self.batches = self._load_batches()
        self._round_robin()
        self.batches = self.batches[self.offset:]
        print(f"[ParquetTextImageDataset] resample done — {len(self.batches)} batches available")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _is_url(path: str) -> bool:
        return path.startswith("http://") or path.startswith("https://")

    @staticmethod
    def scale_and_crop_long_axis(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
        if (target_width / target_height) >= (image.width / image.height):
            image = v2.functional.resize(
                image,
                [round(target_width * image.height / image.width), target_width],
                interpolation=v2.InterpolationMode.LANCZOS,
            )
        else:
            image = v2.functional.resize(
                image,
                [target_height, round(image.width * target_height / image.height)],
                interpolation=v2.InterpolationMode.LANCZOS,
            )
        return v2.functional.center_crop(image, [target_height, target_width])

    @staticmethod
    def _scale_and_crop_tensor(img: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Resize + centre-crop a float ``[C, H, W]`` tensor without quantising to uint8.

        Uses bilinear interpolation (LANCZOS is not available for float tensors
        in torchvision, but bilinear is lossless enough for HDR data).
        """
        _, h, w = img.shape
        if (target_width / target_height) >= (w / h):
            new_w = target_width
            new_h = round(target_width * h / w)
        else:
            new_h = target_height
            new_w = round(w * target_height / h)
        img = v2.functional.resize(
            img, [new_h, new_w], interpolation=v2.InterpolationMode.BILINEAR, antialias=True
        )
        return v2.functional.center_crop(img, [target_height, target_width])

    @staticmethod
    def _sample_elements_by_percentage(my_list: list, percentage: float) -> list:
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        sample_size = math.ceil(len(my_list) * percentage)
        return random.sample(my_list, sample_size)

    @staticmethod
    def dummy_collate_fn(batch):
        return batch

    def tokenize(self, captions: list[str]) -> "torch.Tensor | None":
        """Tokenize *captions* in the DataLoader worker process.

        Returns a ``[B, max_text_len]`` int64 CPU tensor when a tokenizer is
        configured, or ``None`` when no tokenizer is set (no-op / raw-string
        path stays unchanged).
        """
        if self.tokenizer is None:
            return None
        enc = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return enc["input_ids"]  # [B, max_text_len] int64, on CPU

    def __len__(self) -> int:
        return len(self.batches)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image_data(self, data: bytes, filename: str) -> "Image.Image | torch.Tensor | None":
        """Decode raw image *data* bytes into a PIL Image or (if ``raw_hdr=True``
        and the source is HDR/high-bit-depth) a float32 ``[C, H, W]`` tensor.

        Detection order
        ---------------
        1. Magic-byte sniff → format tag
        2. EXR / HDR (Radiance) → always HDR path
        3. TIFF → HDR path (handles float32, uint16, and uint8 TIFFs)
        4. PNG → HDR path only when cv2 reports uint16 depth; 8-bit PNG falls
           through to the standard PIL path
        5. JXL → ``color_profile_handling.open_srgb`` (ICC-managed)
        6. Everything else → plain ``PIL.Image.open``
        """
        fmt = _detect_hdr_format(data[:16])

        # ---- HDR / high-bit-depth formats --------------------------------
        if fmt in ("exr", "hdr"):
            arr = _load_as_float32(data, fmt)
            if self.raw_hdr:
                return torch.from_numpy(arr).permute(2, 0, 1)  # [C,H,W] float32
            return _reinhard_tonemap_to_pil(arr)

        if fmt == "tiff":
            arr = _load_as_float32(data, "tiff")
            # Only treat as HDR if the source was genuinely high-bit-depth
            # (float or uint16+).  8-bit TIFFs are handled normally below.
            if arr.dtype == np.float32 and arr.max() > 1.0:
                # float TIFF with values > 1 → genuine HDR
                if self.raw_hdr:
                    return torch.from_numpy(arr).permute(2, 0, 1)
                return _reinhard_tonemap_to_pil(arr)
            # uint16 or float in [0,1] → normalised to [0,1]; treat as LDR
            img_u8 = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            return Image.fromarray(img_u8)

        if fmt == "png":
            # Peek at bit depth: cv2 ANYDEPTH returns uint16 for 16-bit PNGs.
            buf = np.frombuffer(data, dtype=np.uint8)
            arr_cv2 = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if arr_cv2 is not None and arr_cv2.dtype == np.uint16:
                arr = cv2.cvtColor(arr_cv2, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
                if self.raw_hdr:
                    return torch.from_numpy(arr).permute(2, 0, 1)
                # Normalised to [0,1] linear — apply sRGB gamma only
                srgb = np.where(
                    arr <= 0.0031308,
                    arr * 12.92,
                    1.055 * np.power(np.clip(arr, 0.0, None), 1.0 / 2.4) - 0.055,
                )
                img_u8 = (np.clip(srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                return Image.fromarray(img_u8)
            # 8-bit PNG — fall through to standard PIL path

        # ---- JXL (ICC-managed) -------------------------------------------
        if fmt == "jxl":
            return color_profile_handling.open_srgb(BytesIO(data)).convert("RGB")

        # ---- Standard path (JPEG, WEBP, 8-bit PNG, …) --------------------
        return Image.open(BytesIO(data)).convert("RGB")

    def _read_image_bytes(self, path_or_url: str) -> bytes | None:
        """Return the raw bytes for a local file or URL, or ``None`` on error."""
        try:
            if self._is_url(path_or_url):
                if self.session is None:
                    raise RuntimeError("requests is not installed; cannot load URLs")
                response = self.session.get(path_or_url, timeout=self.timeout)
                response.raise_for_status()
                return response.content
            else:
                with open(path_or_url, "rb") as fh:
                    return fh.read()
        except Exception as e:
            log.error(f"Error reading bytes from '{path_or_url}' on rank {self.rank}: {e}")
            return None

    def _load_image(self, sample: dict) -> "Image.Image | torch.Tensor | None":
        try:
            if self._is_url(sample["filename"]):
                data = self._read_image_bytes(sample["filename"])
                if data is None:
                    return None
                return self._load_image_data(data, sample["filename"])
            else:
                image_path = os.path.join(self.image_folder_path, sample["filename"])
                jxl_path = os.path.splitext(image_path)[0] + ".jxl"
                if os.path.exists(jxl_path):
                    data = self._read_image_bytes(jxl_path)
                    if data is None:
                        return None
                    return self._load_image_data(data, jxl_path)
                elif os.path.exists(image_path):
                    data = self._read_image_bytes(image_path)
                    if data is None:
                        return None
                    return self._load_image_data(data, image_path)
                else:
                    base, _ = os.path.splitext(sample["filename"])
                    for ext in ("png", "jpg", "jpeg", "webp", "exr", "hdr", "tif", "tiff"):
                        alt = os.path.join(self.image_folder_path, f"{base}.{ext}")
                        if os.path.exists(alt):
                            data = self._read_image_bytes(alt)
                            if data is None:
                                continue
                            return self._load_image_data(data, alt)
            return None
        except Exception as e:
            log.error(f"Error loading image '{sample['filename']}' on rank {self.rank}: {e}")
            return None

    def _load_reference_image(self, ref_filename: str) -> "Image.Image | torch.Tensor | None":
        try:
            if self._is_url(ref_filename):
                data = self._read_image_bytes(ref_filename)
                if data is None:
                    return None
                return self._load_image_data(data, ref_filename)
            else:
                image_path = os.path.join(self.image_folder_path, ref_filename)
                jxl_path = os.path.splitext(image_path)[0] + ".jxl"
                if os.path.exists(jxl_path):
                    data = self._read_image_bytes(jxl_path)
                    if data is None:
                        return None
                    return self._load_image_data(data, jxl_path)
                elif os.path.exists(image_path):
                    data = self._read_image_bytes(image_path)
                    if data is None:
                        return None
                    return self._load_image_data(data, image_path)
                else:
                    base, _ = os.path.splitext(ref_filename)
                    for ext in ("png", "jpg", "jpeg", "webp", "exr", "hdr", "tif", "tiff"):
                        alt = os.path.join(self.image_folder_path, f"{base}.{ext}")
                        if os.path.exists(alt):
                            data = self._read_image_bytes(alt)
                            if data is None:
                                continue
                            return self._load_image_data(data, alt)
            return None
        except Exception as e:
            log.error(f"Error loading reference image '{ref_filename}' on rank {self.rank}: {e}")
            return None

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        batch = self.batches[index]

        if not self.dummy_image:
            with _cf.ThreadPoolExecutor() as executor:
                image_futures = [executor.submit(self._load_image, s) for s in batch]

                ref_futures_per_sample = []
                if self.num_reference_images is not None:
                    for sample in batch:
                        ref_list = sample.get("reference_images", [])[: self.num_reference_images]
                        ref_futures_per_sample.append(
                            [executor.submit(self._load_reference_image, r) for r in ref_list]
                        )

                raw_images = [f.result() for f in image_futures]
                raw_reference_images = (
                    [[f.result() for f in futs] for futs in ref_futures_per_sample]
                    if self.num_reference_images is not None
                    else []
                )
        else:
            raw_images = [None] * len(batch)
            raw_reference_images = [[] for _ in batch] if self.num_reference_images is not None else []

        images = []
        training_prompts = []
        loss_weighting = []
        reference_images_batch = []
        tag_strings = []

        for i, sample in enumerate(batch):
            try:
                target_width, target_height = sample["bucket"]

                if not self.dummy_image:
                    img = raw_images[i]
                    if img is None:
                        continue
                    if isinstance(img, torch.Tensor):
                        # Raw HDR tensor [C, H, W] — resize/crop in float space
                        # to preserve full precision (no byte quantisation).
                        img = self._scale_and_crop_tensor(img, target_height, target_width)
                        images.append(img)
                    else:
                        img = self.scale_and_crop_long_axis(img, target_height, target_width)
                        images.append(self.image_transforms(img))
                else:
                    images.append(torch.zeros(3, target_height, target_width))

                # Reference images
                if self.num_reference_images is not None:
                    ref_tensors = []
                    if not self.dummy_image:
                        for ref_img in (raw_reference_images[i] if i < len(raw_reference_images) else []):
                            if ref_img is None:
                                continue
                            if isinstance(ref_img, torch.Tensor):
                                ref_img = self._scale_and_crop_tensor(ref_img, target_height, target_width)
                                ref_tensors.append(ref_img)
                            else:
                                ref_img = self.scale_and_crop_long_axis(ref_img, target_height, target_width)
                                ref_tensors.append(self.image_transforms(ref_img))
                    while len(ref_tensors) < self.num_reference_images:
                        ref_tensors.append(torch.zeros(3, target_height, target_width))
                    reference_images_batch.append(torch.stack(ref_tensors, dim=0))

                # Caption processing
                caption = sample["caption_or_tags"]

                # Unconditional dropout
                if random.random() >= 1 - self.uncond_percentage:
                    caption = ""

                if self.shuffle_tags and sample["is_tag_based"] and caption:
                    tags = caption.split(",")
                    random.shuffle(tags)
                    tags = self._sample_elements_by_percentage(
                        tags, random.uniform(1 - self.tag_drop_percentage, 1)
                    )
                    caption = ",".join(tags).lstrip()

                training_prompts.append(caption)
                loss_weighting.append(sample.get("loss_weight", 1.0))
                tag_strings.append(sample.get("tags", ""))

            except Exception as e:
                log.error(f"Error processing sample '{sample['filename']}' on rank {self.rank}: {e}")
                continue

        # Empty batch fallback
        if len(images) < 1:
            log.info("Empty batch caught — fetching a random replacement batch.")
            return self.__getitem__(random.randrange(0, len(self.batches)))

        # Echo short batches
        if self.rank_batch_size > 1:
            while len(images) < self.rank_batch_size:
                log.info(f"Short batch ({len(images)}/{self.rank_batch_size}), echoing.")
                idx = random.randrange(len(images))
                images.append(images[idx])
                training_prompts.append(training_prompts[idx])
                loss_weighting.append(loss_weighting[idx])
                tag_strings.append(tag_strings[idx])
                if self.num_reference_images is not None and reference_images_batch:
                    reference_images_batch.append(reference_images_batch[idx])

        images = torch.stack(images, dim=0)

        # Tokenize in the worker process so the GPU-thread forward pass gets
        # pre-computed token ids instead of raw strings.
        text_out = self.tokenize(training_prompts)
        # text_out is a Tensor[B, L] when a tokenizer is configured, else None
        # (in which case we fall back to returning the raw string list).
        captions_out = text_out if text_out is not None else training_prompts

        out = [images, captions_out, index, loss_weighting]
        if self.num_reference_images is not None and reference_images_batch:
            out.append(torch.stack(reference_images_batch, dim=0))
        if self.tag_matcher is not None:
            # ONE trailing slot regardless of whether reference images are also
            # enabled, so consumers can read batch[-1] and everything that
            # unpacks batch[:4] keeps working.
            out.append(self._encode_tags(tag_strings))
        return tuple(out)

    def _encode_tags(self, tag_strings: list[str]) -> dict:
        """-> {"tag_ids": [B, max_tags] int64, "tag_mask": [B, max_tags] bool}."""
        b = len(tag_strings)
        ids = np.zeros((b, self.max_tags), dtype=np.int64)
        mask = np.zeros((b, self.max_tags), dtype=bool)
        for i, s in enumerate(tag_strings):
            if not s:
                continue                       # untagged row: exact no-op
            hits = self.tag_matcher.match(s, free_text=False)
            if len(hits) > self.max_tags:
                # Which tags survive an overflow should not depend on the
                # annotator's sort order (this corpus is ~88% alphabetical).
                hits = random.sample(hits, self.max_tags)
            if hits:
                ids[i, : len(hits)] = hits
                mask[i, : len(hits)] = True
        return {
            "tag_ids": torch.from_numpy(ids),
            "tag_mask": torch.from_numpy(mask),
        }


"""Image fetching for the real-image streams (dataloaders/caption_image_stream.py).

``fetch_image`` takes either an http(s) URL or a NAS ``file_path`` and never raises: the result is
``(pixels [3, H, W] float in [-1, 1], None)`` or ``(None, failure reason)``, the reason being a short key
the stream counts in its ``stats``. The image is converted to sRGB (alpha composited on white), rejected
when its short side is below ``min_side``, then resized to cover ``width x height`` and centre-cropped.

NAS paths are read from the local rustfs S3 mirror (point a stream's ``url_column`` at ``file_path``).
trainer_samples_v3 stores them in three forms, all mapped to bucket ``<source>`` / key ``aa/bb/<name>``:

    /mnt/nas_buckets/e621/aa/bb/<hash>.png      e621/aa/bb/<hash>      nas_buckets/deviantart/aa/bb/<id>

The last two carry no extension (and are the rows whose Backblaze URLs 404), so a missing extension is
resolved by probing ``EXTENSIONS`` in order. Endpoint and credentials default to the repo's read-only
rustfs account; override with ``RUSTFS_ENDPOINT`` / ``RUSTFS_CREDENTIAL_FILE``.
"""
from __future__ import annotations

import json
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from dataloaders.color_profile_handling import open_srgb

NAS_PREFIXES = ("/mnt/nas_buckets/", "nas_buckets/")
EXTENSIONS = (".jpg", ".png", ".webp", ".gif", ".jpeg", ".jxl")
_REPO = Path(__file__).resolve().parents[1]

_local = threading.local()
_s3_lock = threading.Lock()
_s3 = None


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def _s3_client():
    global _s3
    with _s3_lock:
        if _s3 is None:
            import boto3
            from botocore.config import Config
            cred_file = os.environ.get("RUSTFS_CREDENTIAL_FILE", str(_REPO / "credential/local-rustfs-readonly.json"))
            cred = json.loads(Path(cred_file).read_text())
            _s3 = boto3.client(
                "s3", endpoint_url=os.environ.get("RUSTFS_ENDPOINT", "http://localhost:9000"),
                aws_access_key_id=cred.get("username", cred.get("access_key")),
                aws_secret_access_key=cred.get("password", cred.get("secret_key")),
                config=Config(signature_version="s3v4", retries={"max_attempts": 2}, max_pool_connections=64))
        return _s3


def nas_object(path: str) -> tuple[str, str]:
    """NAS file_path (any of the three stored forms) -> (bucket, key)."""
    for prefix in NAS_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    bucket, _, key = path.partition("/")
    if not bucket or not key:
        raise ValueError(f"not a NAS path: {path!r}")
    return bucket, key


def read_nas(path: str, retries: int) -> bytes:
    from botocore.exceptions import ClientError
    bucket, key = nas_object(path)
    name = key.rsplit("/", 1)[-1]
    candidates = [key] if "." in name else [key + ext for ext in EXTENSIONS]
    client = _s3_client()
    for candidate in candidates:
        for attempt in range(retries + 1):
            try:
                return client.get_object(Bucket=bucket, Key=candidate)["Body"].read()
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NoSuchBucket"):
                    break
                if attempt == retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
    raise FileNotFoundError(f"{bucket}/{key} (tried {len(candidates)} key(s))")


def download(url: str, timeout: float, retries: int) -> bytes:
    for attempt in range(retries + 1):
        try:
            response = _session().get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")


def centre_crop(image: Image.Image, width: int, height: int) -> torch.Tensor:
    scale = max(width / image.width, height / image.height)
    size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    image = image.crop((left, top, left + width, top + height))
    array = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
    return array.float().div(127.5).sub(1.0)


def fetch_image(source: str, width: int, height: int, min_side: int, timeout: float = 30.0,
                retries: int = 2) -> tuple[torch.Tensor | None, str | None]:
    try:
        if source.startswith(("http://", "https://")):
            data = download(source, timeout, retries)
        else:
            data = read_nas(source, retries)
    except (requests.RequestException, FileNotFoundError, ValueError):
        return None, "fetch_failed"
    except Exception:
        return None, "fetch_error"
    try:
        image = open_srgb(BytesIO(data))
        if image.mode == "RGBA":
            image = Image.alpha_composite(Image.new("RGBA", image.size, (255, 255, 255, 255)), image)
        image = image.convert("RGB")
    except Exception:
        return None, "decode_failed"
    if min(image.size) < min_side:
        return None, "too_small"
    return centre_crop(image, width, height), None

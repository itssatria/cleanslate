"""Model file management: check whether a model exists locally, download it if not.

LaMa weights are stored where simple-lama-inpainting looks for them
(<torch hub dir>/checkpoints/big-lama.pt), so the library finds the file and
never needs to download anything itself. Set LAMA_MODEL to a custom file path
to use your own copy, or LAMA_MODEL_URL to add your own download mirror.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

LAMA_FILENAME = "big-lama.pt"
LAMA_URLS = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)
LAMA_MIN_SIZE = 150 * 1024 * 1024  # real file is ~200 MB; smaller means a broken download

ProgressFn = Callable[[float, str], None]


class ModelDownloadError(Exception):
    pass


def lama_path() -> Path:
    """Where the LaMa weights live (or will be downloaded to)."""
    custom = os.environ.get("LAMA_MODEL")
    if custom:
        return Path(custom)
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / LAMA_FILENAME


def lama_is_downloaded(path: Optional[Path] = None, min_size: int = LAMA_MIN_SIZE) -> bool:
    p = path or lama_path()
    return p.is_file() and p.stat().st_size >= min_size


def _download(url: str, dest: Path, progress: Optional[ProgressFn], min_size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "wmremover"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total, f"Downloading LaMa model (first run only): {done >> 20} / {total >> 20} MB")
        if tmp.stat().st_size < min_size:
            raise OSError("download incomplete")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def ensure_lama_model(
    progress: Optional[ProgressFn] = None,
    *,
    dest: Optional[Path] = None,
    urls: Optional[Sequence[str]] = None,
    min_size: int = LAMA_MIN_SIZE,
) -> Path:
    """Return the path to the LaMa weights, downloading them first if missing."""
    dest = dest or lama_path()
    if lama_is_downloaded(dest, min_size):
        return dest
    if dest.exists():  # truncated or corrupt file from an earlier attempt
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)

    candidates = list(urls or ())
    if not candidates:
        extra = os.environ.get("LAMA_MODEL_URL")
        candidates = ([extra] if extra else []) + list(LAMA_URLS)

    errors = []
    for url in candidates:
        try:
            if progress:
                progress(0.0, "Downloading LaMa model (first run only)...")
            _download(url, dest, progress, min_size)
            return dest
        except Exception as e:
            errors.append(f"{url}: {e}")

    raise ModelDownloadError(
        "Could not download the LaMa model.\n"
        + "\n".join(errors)
        + f"\n\nDownload big-lama.pt manually and place it at:\n{dest}"
    )

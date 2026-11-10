"""Model file management: look for a model locally, download it if it is missing.

LaMa weights (big-lama.pt, ~200 MB) are searched for in this order:
  1. the path in the LAMA_MODEL environment variable
  2. Cleanslate's own folder  (%LOCALAPPDATA%\\cleanslate\\models on Windows, ~/.cache/cleanslate/models elsewhere)
  3. the PyTorch hub cache    (~/.cache/torch/hub/checkpoints)  - so an existing copy is reused
If none exists, the file is downloaded to (1) or (2). Add your own mirror with LAMA_MODEL_URL.
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
LAMA_MIN_SIZE = 150 * 1024 * 1024  # the real file is ~200 MB; smaller means a broken download

ProgressFn = Callable[[float, str], None]


class ModelDownloadError(Exception):
    pass


def models_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "cleanslate" / "models"


def _torch_hub_copy() -> Path:
    home = os.environ.get("TORCH_HOME") or str(Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "torch")
    return Path(home) / "hub" / "checkpoints" / LAMA_FILENAME


def default_lama_path() -> Path:
    custom = os.environ.get("LAMA_MODEL")
    return Path(custom) if custom else models_dir() / LAMA_FILENAME


def lama_is_downloaded(path: Path, min_size: int = LAMA_MIN_SIZE) -> bool:
    return path.is_file() and path.stat().st_size >= min_size


def find_lama(min_size: int = LAMA_MIN_SIZE) -> Optional[Path]:
    for p in (default_lama_path(), models_dir() / LAMA_FILENAME, _torch_hub_copy()):
        if lama_is_downloaded(p, min_size):
            return p
    return None


def _download(url: str, dest: Path, progress: Optional[ProgressFn], min_size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cleanslate"})
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
    if dest is None:
        found = find_lama(min_size)
        if found:
            return found
        dest = default_lama_path()
    elif lama_is_downloaded(dest, min_size):
        return dest

    if dest.exists():  # truncated/corrupt file from an earlier attempt
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

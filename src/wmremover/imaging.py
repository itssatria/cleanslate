"""Image helpers: folder scan, thumbnails, mask handling, compositing, saving.

No Gradio imports here so this module stays easy to test.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
THUMB_SIZE = 256
THUMB_DIR = Path(tempfile.gettempdir()) / "wmremover_thumbs"


def list_images(folder: str) -> list[Path]:
    folder = (folder or "").strip().strip('"')
    if not folder or not Path(folder).is_dir():
        raise ValueError("Folder not found.")
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise ValueError("No images found in that folder.")
    return files


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def make_thumb(path: Path, size: int = THUMB_SIZE) -> str:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{path}{path.stat().st_mtime}{size}".encode()).hexdigest()
    out = THUMB_DIR / f"{key}.jpg"
    if not out.exists():
        im = load_image(path)
        im.thumbnail((size, size))
        im.save(out, quality=85)
    return str(out)


def extract_mask(layers: Iterable, size: tuple[int, int]) -> np.ndarray:
    """Union of all painted layers -> uint8 mask (255 = remove), shape (h, w)."""
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    for layer in layers or []:
        if layer is None:
            continue
        arr = np.array(layer.convert("RGBA"))
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        mask |= (arr[..., 3] > 10).astype(np.uint8) * 255
    if not mask.any():
        raise ValueError("Mask is empty - paint over the watermark first.")
    return mask


def grow_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, k)


def paste_masked(orig: Image.Image, result: Image.Image, mask: np.ndarray, feather: float = 3) -> Image.Image:
    """Replace only the masked region (with a soft edge) so the rest stays pixel-identical."""
    result = result.convert("RGB")
    if result.size != orig.size:
        result = result.resize(orig.size, Image.LANCZOS)
    soft = cv2.GaussianBlur(mask, (0, 0), feather) if feather > 0 else mask
    soft = np.maximum(soft, mask).astype(np.float32)[..., None] / 255.0
    out = np.asarray(orig, np.float32) * (1 - soft) + np.asarray(result, np.float32) * soft
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def save_result(result: Image.Image, src: str | Path, out_dir_name: str = "cleaned") -> Path:
    src = Path(src)
    out_dir = src.parent / out_dir_name
    out_dir.mkdir(exist_ok=True)
    out = out_dir / src.name
    if src.suffix.lower() in (".jpg", ".jpeg"):
        result.save(out, quality=95, subsampling=0)
    else:
        result.save(out)
    return out

"""LaMa inpainting, run directly from the TorchScript weights.

Cleanslate loads big-lama.pt itself with torch.jit.load (always mapped to CPU first, then moved to the
chosen device), so it only needs `torch`: no torchvision and no simple-lama-inpainting.
Override the device with CLEANSLATE_DEVICE=cpu|cuda.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ..models import ModelDownloadError, ensure_lama_model
from .base import Backend, BackendError

MAX_SIDE = 1600  # crops larger than this are downscaled for inference (keeps CPU runs reasonable)


def crop_box(mask: np.ndarray, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Box (x0, y0, x1, y1) around the mask plus context, clamped to the image. size = (w, h)."""
    ys, xs = np.nonzero(mask)
    span = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    pad = max(128, span // 2)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, size[0])
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, size[1])
    return x0, y0, x1, y1


def pad_to_multiple(img: np.ndarray, mask: np.ndarray, mod: int = 8):
    """Pad bottom/right so height and width are multiples of `mod` (LaMa requirement)."""
    h, w = mask.shape
    ph, pw = (-h) % mod, (-w) % mod
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="edge")
    return img, mask


class LamaBackend(Backend):
    name = "LaMa (local)"
    feather = 2

    def __init__(self):
        self._model = None
        self._device = None

    # -- model management -------------------------------------------------
    def ensure_ready(self, progress=None):
        """Check for the model file; download it automatically on first use."""
        try:
            ensure_lama_model(progress)
        except ModelDownloadError as e:
            raise BackendError(str(e)) from e

    def _load(self, device: str):
        try:
            import torch
        except ImportError as e:
            raise BackendError("PyTorch is not installed. Run: pip install torch") from e
        try:
            path = ensure_lama_model()
            model = torch.jit.load(str(path), map_location="cpu")
            model.eval()
            self._model = model.to(device)
        except ModelDownloadError as e:
            raise BackendError(str(e)) from e
        except Exception as e:
            raise BackendError(f"LaMa failed to load: {e}") from e
        self._device = device

    def _pick_device(self) -> str:
        device = os.environ.get("CLEANSLATE_DEVICE")
        if device:
            return device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    # -- inference --------------------------------------------------------
    def _infer(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        import torch

        w, h = image.size
        scale = min(1.0, MAX_SIDE / max(w, h))
        if scale < 1.0:
            sw, sh = max(8, round(w * scale)), max(8, round(h * scale))
            image = image.resize((sw, sh), Image.LANCZOS)
            mask = np.asarray(Image.fromarray(mask).resize((sw, sh), Image.NEAREST))

        img = np.asarray(image, np.float32) / 255.0
        m = (mask > 0).astype(np.float32)
        sh, sw = m.shape
        img, m = pad_to_multiple(img, m)

        t_img = torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0).to(self._device)
        t_mask = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            out = self._model(t_img, t_mask)
        out = out[0].permute(1, 2, 0).float().cpu().numpy()[:sh, :sw]
        result = Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))
        return result.resize((w, h), Image.LANCZOS) if scale < 1.0 else result

    def _run(self, image, mask):
        if self._model is None:
            self._load(self._pick_device())
        try:
            return self._infer(image, mask)
        except (NotImplementedError, RuntimeError) as e:
            if self._device == "cpu":
                raise BackendError(f"LaMa failed on CPU: {e}") from e
            self._load("cpu")  # GPU op missing / out of memory: retry on CPU
            return self._infer(image, mask)

    def inpaint(self, image, mask, *, api_key=None, prompt=None):
        # Work on a crop around the mask instead of the whole image: much faster on CPU.
        x0, y0, x1, y1 = crop_box(mask, image.size)
        crop_out = self._run(image.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1])
        result = image.copy()
        result.paste(crop_out.convert("RGB"), (x0, y0))
        return result

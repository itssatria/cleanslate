from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ..models import ModelDownloadError, ensure_lama_model
from .base import Backend, BackendError


class LamaBackend(Backend):
    name = "LaMa (local)"
    feather = 2

    def __init__(self):
        self._model = None
        self._device = None

    def ensure_ready(self, progress=None):
        """Check for the model file; download it automatically on first use."""
        try:
            ensure_lama_model(progress)
        except ModelDownloadError as e:
            raise BackendError(str(e)) from e
        except ImportError as e:
            raise BackendError('PyTorch is not installed. Run: pip install "wmremover[lama]"') from e

    def _load(self, device: str):
        try:
            import torch
            from simple_lama_inpainting import SimpleLama

            self._model = SimpleLama(device=torch.device(device))
        except ImportError as e:
            raise BackendError('LaMa is not installed. Run: pip install "wmremover[lama]"') from e
        except Exception as e:
            raise BackendError(f"LaMa failed to load: {e}") from e
        self._device = device

    def _run(self, image, mask):
        if self._model is None:
            device = os.environ.get("WMR_DEVICE")
            if not device:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._load(device)
        try:
            return self._model(image, Image.fromarray(mask))
        except (NotImplementedError, RuntimeError) as e:
            if self._device == "cpu":
                raise BackendError(f"LaMa failed on CPU: {e}") from e
            self._load("cpu")
            return self._model(image, Image.fromarray(mask))

    def inpaint(self, image, mask, *, api_key=None, prompt=None):
        # Run on a crop around the mask (plus context) instead of the whole image: much faster on CPU.
        ys, xs = np.nonzero(mask)
        pad = max(128, int(max(xs.max() - xs.min(), ys.max() - ys.min())) // 2)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, image.width)
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, image.height)

        crop_out = self._run(image.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1])
        if crop_out.size != (x1 - x0, y1 - y0):
            crop_out = crop_out.resize((x1 - x0, y1 - y0), Image.LANCZOS)

        result = image.copy()
        result.paste(crop_out.convert("RGB"), (x0, y0))
        return result

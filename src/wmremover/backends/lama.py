from __future__ import annotations

from PIL import Image

from .base import Backend, BackendError


class LamaBackend(Backend):
    name = "LaMa (local)"
    feather = 2

    def __init__(self):
        self._model = None

    def inpaint(self, image, mask, *, api_key=None, prompt=None):
        if self._model is None:
            try:
                from simple_lama_inpainting import SimpleLama
            except ImportError as e:
                raise BackendError('LaMa is not installed. Run: pip install "wmremover[lama]"') from e
            self._model = SimpleLama()
        return self._model(image, Image.fromarray(mask))

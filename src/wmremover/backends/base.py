from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class BackendError(Exception):
    """User-facing backend failure (missing package, missing key, API error...)."""


class Backend(ABC):
    #: label shown in the UI
    name: str = "base"
    #: paste only the masked region back onto the original after inpainting
    composite: bool = True
    #: edge softness used when compositing
    feather: float = 3
    #: show API key / prompt fields for this backend
    uses_api: bool = False

    @abstractmethod
    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        *,
        api_key: str | None = None,
        prompt: str | None = None,
    ) -> Image.Image:
        """image: RGB PIL. mask: uint8 (h, w), 255 = area to remove. Return RGB PIL."""

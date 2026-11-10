from __future__ import annotations

import base64
import io
import os

import cv2
import numpy as np
from PIL import Image

from .base import Backend, BackendError

DEFAULT_PROMPT = (
    "Remove the watermark and restore the background naturally. "
    "Do not change anything else in the image."
)


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class OpenAIBackend(Backend):
    name = "OpenAI API"
    feather = 5
    uses_api = True

    def inpaint(self, image, mask, *, api_key=None, prompt=None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise BackendError('Run: pip install "cleanslate[openai]"') from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise BackendError("Enter an OpenAI API key (or set OPENAI_API_KEY).")

        w, h = image.size
        ratio = w / h
        size = (1536, 1024) if ratio > 1.2 else (1024, 1536) if ratio < 0.83 else (1024, 1024)
        small = image.resize(size, Image.LANCZOS)
        m_small = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        rgba = np.full((size[1], size[0], 4), 255, np.uint8)
        rgba[m_small > 0, 3] = 0  # transparent = area the model may edit

        try:
            res = OpenAI(api_key=key).images.edit(
                model="gpt-image-1",
                image=("image.png", _png(small), "image/png"),
                mask=("mask.png", _png(Image.fromarray(rgba, "RGBA")), "image/png"),
                prompt=prompt or DEFAULT_PROMPT,
                size=f"{size[0]}x{size[1]}",
            )
        except Exception as e:  # network, auth, policy, quota...
            raise BackendError(f"OpenAI request failed: {e}") from e
        return Image.open(io.BytesIO(base64.b64decode(res.data[0].b64_json)))

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .base import Backend


class OpenCVBackend(Backend):
    name = "OpenCV (fast, small marks)"
    composite = False

    def inpaint(self, image, mask, *, api_key=None, prompt=None):
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

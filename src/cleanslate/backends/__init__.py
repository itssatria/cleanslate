"""Backend registry. To add a backend: subclass Backend, then register it below."""
from .base import Backend, BackendError
from .lama import LamaBackend
from .opencv_telea import OpenCVBackend
from .openai_api import DEFAULT_PROMPT, OpenAIBackend

BACKENDS: dict[str, Backend] = {b.name: b for b in (LamaBackend(), OpenCVBackend(), OpenAIBackend())}

__all__ = ["BACKENDS", "Backend", "BackendError", "DEFAULT_PROMPT"]

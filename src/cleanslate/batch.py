"""Batch processing: apply one reference mask to every image in a list.

Pure Python (no Gradio): run_batch() is a generator that yields BatchEvent objects, so the UI can
show progress and tests can drive it with a fake backend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

from .imaging import grow_mask, load_image, output_path, paste_masked, save_result, scale_mask


@dataclass
class BatchEvent:
    index: int  # 0-based position in the file list
    total: int
    path: Path
    status: str  # "running" | "done" | "skipped" | "failed"
    message: str = ""
    seconds: float = 0.0


def run_batch(
    files: Sequence[str | Path],
    ref_mask: np.ndarray,
    backend,
    *,
    grow_px: int = 6,
    on_size_mismatch: str = "scale",  # "scale" | "skip"
    on_existing: str = "skip",  # "skip" | "overwrite"
    api_key: Optional[str] = None,
    prompt: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Iterator[BatchEvent]:
    """Clean every file with `ref_mask` (uint8, 255 = remove, sized like the reference image).

    One image failing never stops the run: it is reported as "failed" and the loop continues.
    """
    total = len(files)
    ref_h, ref_w = ref_mask.shape
    for i, raw in enumerate(files):
        if should_stop and should_stop():
            return
        path = Path(raw)
        if on_existing == "skip" and output_path(path).exists():
            yield BatchEvent(i, total, path, "skipped", "already cleaned")
            continue

        yield BatchEvent(i, total, path, "running")
        t0 = time.perf_counter()
        try:
            img = load_image(path)
            if img.size != (ref_w, ref_h) and on_size_mismatch == "skip":
                yield BatchEvent(i, total, path, "skipped", "size differs from reference")
                continue
            mask = scale_mask(ref_mask, img.size)
            mask = grow_mask(mask, round(grow_px * img.width / ref_w))
            out = backend.inpaint(img, mask, api_key=api_key, prompt=prompt)
            if backend.composite:
                out = paste_masked(img, out, mask, feather=backend.feather)
            save_result(out, path)
        except Exception as e:  # keep going: report and move on
            yield BatchEvent(i, total, path, "failed", str(e), time.perf_counter() - t0)
        else:
            yield BatchEvent(i, total, path, "done", "", time.perf_counter() - t0)

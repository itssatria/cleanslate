import numpy as np
from PIL import Image

from cleanslate.batch import run_batch
from cleanslate.imaging import output_path


class FakeBackend:
    """Paints the whole image red; the batch code must paste only the masked area back."""
    composite = True
    feather = 0

    def __init__(self, fail_on=None):
        self.fail_on = fail_on

    def inpaint(self, image, mask, *, api_key=None, prompt=None):
        if self.fail_on and image.size == self.fail_on:
            raise RuntimeError("boom")
        return Image.new("RGB", image.size, (255, 0, 0))


def _make(tmp_path):
    big1, big2, small = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    for p, size in ((big1, (40, 30)), (big2, (40, 30)), (small, (20, 15))):
        Image.new("RGB", size, (0, 0, 255)).save(p)
    mask = np.zeros((30, 40), np.uint8)
    mask[20:28, 28:38] = 255  # bottom-right "watermark"
    return [big1, big2, small], mask


def _final(events):
    return {e.path.name: e.status for e in events if e.status != "running"}


def test_scale_mode_cleans_every_size(tmp_path):
    files, mask = _make(tmp_path)
    ev = list(run_batch(files, mask, FakeBackend(), grow_px=0))
    assert _final(ev) == {"a.png": "done", "b.png": "done", "c.png": "done"}
    out = np.array(Image.open(output_path(files[0])))
    assert tuple(out[24, 33]) == (255, 0, 0)   # inside mask: replaced
    assert tuple(out[2, 2]) == (0, 0, 255)     # outside mask: untouched
    small = Image.open(output_path(files[2]))
    assert small.size == (20, 15)              # scaled mask, original size kept
    assert tuple(np.array(small)[12, 16]) == (255, 0, 0)


def test_skip_mode_skips_other_sizes(tmp_path):
    files, mask = _make(tmp_path)
    ev = list(run_batch(files, mask, FakeBackend(), grow_px=0, on_size_mismatch="skip"))
    assert _final(ev)["c.png"] == "skipped"
    assert not output_path(files[2]).exists()


def test_existing_outputs_are_skipped_then_overwritten(tmp_path):
    files, mask = _make(tmp_path)
    list(run_batch(files, mask, FakeBackend(), grow_px=0))
    second = _final(run_batch(files, mask, FakeBackend(), grow_px=0))
    assert set(second.values()) == {"skipped"}
    third = _final(run_batch(files, mask, FakeBackend(), grow_px=0, on_existing="overwrite"))
    assert set(third.values()) == {"done"}


def test_one_failure_does_not_stop_the_run(tmp_path):
    files, mask = _make(tmp_path)
    ev = _final(run_batch(files, mask, FakeBackend(fail_on=(20, 15)), grow_px=0))
    assert ev == {"a.png": "done", "b.png": "done", "c.png": "failed"}


def test_stop_callback_ends_run_early(tmp_path):
    files, mask = _make(tmp_path)
    seen = []

    def stop():
        return len(seen) >= 1

    for e in run_batch(files, mask, FakeBackend(), grow_px=0, should_stop=stop):
        if e.status == "done":
            seen.append(e.path.name)
    assert seen == ["a.png"]

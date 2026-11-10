import numpy as np
import pytest
from PIL import Image

from wmremover.imaging import extract_mask, grow_mask, list_images, paste_masked, save_result


def _layer(size, box):
    arr = np.zeros((size[1], size[0], 4), np.uint8)
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = (255, 0, 0, 255)
    return Image.fromarray(arr, "RGBA")


def test_extract_mask_union():
    m = extract_mask([_layer((20, 10), (0, 0, 5, 5)), _layer((20, 10), (10, 5, 15, 10))], (20, 10))
    assert m.shape == (10, 20)
    assert m[0, 0] == 255 and m[9, 14] == 255 and m[0, 19] == 0


def test_extract_mask_empty_raises():
    with pytest.raises(ValueError):
        extract_mask([], (10, 10))


def test_grow_mask_expands():
    m = np.zeros((21, 21), np.uint8)
    m[10, 10] = 255
    assert grow_mask(m, 3).sum() > m.sum()
    assert (grow_mask(m, 0) == m).all()


def test_paste_masked_keeps_outside_pixels():
    orig = Image.new("RGB", (40, 40), (10, 20, 30))
    res = Image.new("RGB", (40, 40), (200, 200, 200))
    mask = np.zeros((40, 40), np.uint8)
    mask[15:25, 15:25] = 255
    out = np.array(paste_masked(orig, res, mask, feather=0))
    assert tuple(out[0, 0]) == (10, 20, 30)
    assert tuple(out[20, 20]) == (200, 200, 200)


def test_list_images(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in list_images(str(tmp_path))] == ["a.png"]
    with pytest.raises(ValueError):
        list_images(str(tmp_path / "missing"))


def test_save_result_goes_to_cleaned(tmp_path):
    src = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(src)
    out = save_result(Image.new("RGB", (4, 4)), src)
    assert out == tmp_path / "cleaned" / "a.png" and out.exists()

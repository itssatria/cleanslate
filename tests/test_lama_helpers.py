import numpy as np

from cleanslate.backends.lama import crop_box, pad_to_multiple


def test_pad_to_multiple_of_8():
    img = np.zeros((13, 21, 3), np.float32)
    mask = np.zeros((13, 21), np.float32)
    i2, m2 = pad_to_multiple(img, mask)
    assert i2.shape[:2] == m2.shape == (16, 24)


def test_pad_noop_when_aligned():
    img = np.zeros((16, 24, 3), np.float32)
    mask = np.zeros((16, 24), np.float32)
    assert pad_to_multiple(img, mask)[1].shape == (16, 24)


def test_crop_box_contains_mask_and_is_clamped():
    mask = np.zeros((1000, 2000), np.uint8)
    mask[900:990, 1900:1990] = 255
    x0, y0, x1, y1 = crop_box(mask, (2000, 1000))
    assert x0 <= 1900 and y0 <= 900 and x1 == 2000 and y1 == 1000
    assert x0 >= 0 and y0 >= 0

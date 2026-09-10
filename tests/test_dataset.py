"""Unit tests for box-prompt utilities."""
import numpy as np

from src.data.dataset import jitter_box, mask_to_box


def test_mask_to_box():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 20:60] = 1
    box = mask_to_box(mask, margin=5)
    assert box == [15.0, 25.0, 64.0, 74.0]


def test_mask_to_box_margin_clipped_to_bounds():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[0:5, 0:5] = 1  # touches the top-left corner
    box = mask_to_box(mask, margin=10)
    x0, y0, x1, y1 = box
    assert x0 >= 0
    assert y0 >= 0


def test_jitter_box_stays_in_bounds_and_ordered():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 20:60] = 1
    box = mask_to_box(mask)
    for _ in range(50):
        x0, y0, x1, y1 = jitter_box(box, mask.shape, max_jitter=10)
        assert 0 <= x0 < x1 <= 100
        assert 0 <= y0 < y1 <= 100

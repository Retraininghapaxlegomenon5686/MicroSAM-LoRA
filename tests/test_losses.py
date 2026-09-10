"""Unit tests for loss functions and metrics."""
import numpy as np
import torch

from src.losses import DiceBCELoss, DiceLoss, dice_coefficient, iou_score, keep_largest_component


def test_dice_loss_perfect_match():
    logits = torch.full((2, 1, 4, 4), 10.0)
    targets = torch.ones(2, 1, 4, 4)
    loss = DiceLoss()(logits, targets)
    assert loss.item() < 0.01


def test_dice_loss_total_mismatch():
    logits = torch.full((2, 1, 4, 4), 10.0)
    targets = torch.zeros(2, 1, 4, 4)
    loss = DiceLoss()(logits, targets)
    assert loss.item() > 0.9


def test_dice_bce_loss_perfect_match():
    logits = torch.full((2, 1, 4, 4), 10.0)
    targets = torch.ones(2, 1, 4, 4)
    loss = DiceBCELoss()(logits, targets)
    assert loss.item() < 0.01


def test_dice_bce_loss_total_mismatch_is_large():
    logits = torch.full((2, 1, 4, 4), 10.0)
    targets = torch.zeros(2, 1, 4, 4)
    loss = DiceBCELoss()(logits, targets)
    assert loss.item() > 5.0


def test_dice_coefficient_extremes():
    logits = torch.full((2, 1, 4, 4), 10.0)
    perfect = torch.ones(2, 1, 4, 4)
    mismatch = torch.zeros(2, 1, 4, 4)
    assert dice_coefficient(logits, perfect) > 0.99
    assert dice_coefficient(logits, mismatch) < 0.1


def test_iou_score_extremes():
    logits = torch.full((2, 1, 4, 4), 10.0)
    perfect = torch.ones(2, 1, 4, 4)
    mismatch = torch.zeros(2, 1, 4, 4)
    assert iou_score(logits, perfect) > 0.99
    assert iou_score(logits, mismatch) < 0.1


def test_keep_largest_component_removes_small_blobs():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 1       # small blob, 9 px
    mask[10:18, 10:18] = 1   # large blob, 64 px
    cleaned = keep_largest_component(mask)
    assert cleaned.sum() == 64
    assert cleaned[3, 3] == 0    # small blob removed
    assert cleaned[14, 14] == 1  # large blob kept


def test_keep_largest_component_empty_mask_unchanged():
    mask = np.zeros((20, 20), dtype=np.uint8)
    cleaned = keep_largest_component(mask)
    assert cleaned.sum() == 0

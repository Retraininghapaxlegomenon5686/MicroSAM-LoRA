"""Unit tests for the LoRA implementation."""
import torch
import torch.nn as nn

from src.models.lora import LoRALinear


def test_lora_zero_init_matches_base():
    """At init, LoRA's B matrix is zero, so output should exactly match the
    base linear layer's output before any training happens."""
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_base_params_frozen_new_params_trainable():
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    assert not lora.base.weight.requires_grad
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_lora_device_consistency():
    """Regression test for a real bug hit during development: LoRA params
    must be created on the same device as the base layer, even when
    injected after the parent model has already been moved with .to(device).
    Without this, lora_A/lora_B silently default to CPU and the first
    forward pass raises a device-mismatch RuntimeError."""
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    devices = {p.device for p in lora.parameters()}
    assert len(devices) == 1


def test_lora_forward_shape():
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(5, 16)
    out = lora(x)
    assert out.shape == (5, 32)


def test_lora_gradients_flow_only_through_lora_params():
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 16, requires_grad=False)
    out = lora(x)
    out.sum().backward()
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert lora.base.weight.grad is None  # frozen — must not receive gradients

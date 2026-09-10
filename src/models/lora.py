"""LoRA (Low-Rank Adaptation), implemented from scratch, and injection into
MedSAM's ViT-B vision encoder.

Instead of fine-tuning a full weight matrix W, freeze W entirely and learn a
low-rank *update* B @ A, with rank r tiny (8 by default) compared to the
full dimension (768+). Forward becomes W(x) + scale * B(A(x)).
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update.

    Two details that matter:
    - `lora_B` is zero-initialized, so at step 0 the LoRA path contributes
      nothing — training starts from a state numerically identical to the
      frozen pretrained model, not an immediate perturbation of it.
    - New parameters are created on the *same device as the base layer*
      (not the default CPU), so injecting LoRA after `model.to(device)`
      doesn't leave the new parameters stranded on CPU — a real bug this
      project hit and fixed (see README).
    """

    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base_linear.in_features, base_linear.out_features
        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, in_f, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.lora_A.T) @ self.lora_B.T)


def inject_lora(model, r=8, alpha=16):
    """Replace qkv + proj in every ViT-B attention block with a LoRA-wrapped
    version. HuggingFace's SAM implementation fuses Q/K/V into one 'qkv'
    linear layer per block, plus a separate output 'proj' — verified
    directly against the library before writing this (see README)."""
    for layer in model.vision_encoder.layers:
        layer.attn.qkv = LoRALinear(layer.attn.qkv, r=r, alpha=alpha)
        layer.attn.proj = LoRALinear(layer.attn.proj, r=r, alpha=alpha)
    return model


def prepare_medsam_lora(model, r=8, alpha=16):
    """Freeze everything, inject LoRA into the vision encoder, fully
    unfreeze the mask decoder. The prompt encoder and the rest of the vision
    encoder (patch embed, layer norms, neck) stay frozen."""
    for p in model.parameters():
        p.requires_grad = False
    inject_lora(model, r=r, alpha=alpha)
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
    return model


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

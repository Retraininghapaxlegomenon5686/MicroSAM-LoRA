"""VGG16-UNet: ImageNet-pretrained VGG16 encoder (frozen) + trainable U-Net decoder.

With ~1,700 training slices from 44 patients, training a deep encoder from
scratch invites overfitting fast; reusing ImageNet-pretrained features and
training only the decoder is a much better-conditioned problem. Notably,
this is the same underlying idea as freezing MedSAM's encoder and training
only a LoRA adapter in Phase 2 — a hand-picked "adapter" (the decoder) here,
a learned low-rank one there.
"""
import torch
import torch.nn as nn
import torchvision


class VGG16UNet(nn.Module):
    def __init__(self, pretrained=True, freeze_encoder=True):
        super().__init__()
        vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1" if pretrained else None).features

        # Slice VGG16's feature list at each block boundary (right before each maxpool)
        self.enc1, self.pool1 = vgg[:4], vgg[4]      # -> 64 ch
        self.enc2, self.pool2 = vgg[5:9], vgg[9]      # -> 128 ch
        self.enc3, self.pool3 = vgg[10:16], vgg[16]   # -> 256 ch
        self.enc4, self.pool4 = vgg[17:23], vgg[23]   # -> 512 ch
        self.bottleneck = vgg[24:30]                  # -> 512 ch

        if freeze_encoder:
            for p in self.parameters():
                p.requires_grad = False

        self.up4 = self._decoder_block(512, 512, 512)
        self.up3 = self._decoder_block(512, 256, 256)
        self.up2 = self._decoder_block(256, 128, 128)
        self.up1 = self._decoder_block(128, 64, 64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    @staticmethod
    def _decoder_block(in_ch, skip_ch, out_ch):
        return nn.ModuleDict({
            "upsample": nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            "conv": nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ),
        })

    def _decode(self, block, x, skip):
        x = block["upsample"](x)
        x = torch.cat([x, skip], dim=1)
        return block["conv"](x)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        b = self.bottleneck(self.pool4(s4))

        d4 = self._decode(self.up4, b, s4)
        d3 = self._decode(self.up3, d4, s3)
        d2 = self._decode(self.up2, d3, s2)
        d1 = self._decode(self.up1, d2, s1)
        return self.head(d1)

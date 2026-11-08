"""Independent DINOv2-L satellite tile encoder."""

from contextlib import nullcontext

import torch
import torch.nn as nn

from .dinov2.hub.backbones import dinov2_vitl14_reg


class SatelliteEncoder(nn.Module):
    def __init__(
        self,
        tile_size=518,
        pretrained=True,
        freeze_backbone=True,
    ):
        super().__init__()
        self.tile_size = int(tile_size)
        self.backbone = dinov2_vitl14_reg(pretrained=pretrained, img_size=tile_size)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, tiles):
        parameter = next(self.backbone.parameters())
        tiles = tiles.to(device=parameter.device, dtype=parameter.dtype)
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            patch_tokens = self.backbone(tiles, is_training=True)["x_norm_patchtokens"]

        return {"patch_tokens": patch_tokens}

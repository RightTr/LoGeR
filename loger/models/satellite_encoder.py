"""Independent DINOv2-S satellite tile encoder."""

from contextlib import nullcontext

import torch
import torch.nn as nn

from .dinov2.hub.backbones import dinov2_vits14_reg


class SatelliteEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        tile_size=518,
        pool_size=16,
        pretrained=True,
        freeze_backbone=True,
    ):
        super().__init__()
        self.tile_size = int(tile_size)
        self.pool_size = int(pool_size)
        self.backbone = dinov2_vits14_reg(pretrained=pretrained, img_size=tile_size)
        self.pooling = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.projection = nn.Linear(self.backbone.embed_dim, embed_dim)
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

        batch_size, _, channels = patch_tokens.shape
        grid_size = self.backbone.patch_embed.patches_resolution[0]
        patch_grid = patch_tokens.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        pooled_grid = self.pooling(patch_grid)
        pooled_tokens = pooled_grid.flatten(2).transpose(1, 2)
        projected_tokens = self.projection(pooled_tokens)
        return {
            "patch_tokens": patch_tokens,
            "pool_tokens": pooled_tokens,
            "projected_tokens": projected_tokens,
        }

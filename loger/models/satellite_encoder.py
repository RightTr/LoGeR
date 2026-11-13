"""Encode full DOP images into overlap-merged frozen DINOv2-S patch maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from PIL import Image

from .dinov2.hub.backbones import dinov2_vits14_reg


class SatelliteEncoder(nn.Module):
    """Frozen DINOv2-S/14-Reg with tiled full-map inference and feathered blending."""

    patch_size = 14
    embed_dim = 384
    tile_size = 518
    tile_stride = 400
    output_grid = 128

    def __init__(self, pretrained: bool = True, tile_batch_size: int = 4):
        super().__init__()
        self.backbone = dinov2_vits14_reg(pretrained=pretrained)
        del self.backbone.mask_token
        self.tile_batch_size = int(tile_batch_size)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    @staticmethod
    def _starts(length: int, tile: int, stride: int) -> list[int]:
        if length <= tile:
            return [0]
        starts = list(range(0, length - tile + 1, stride))
        final = length - tile
        if starts[-1] != final:
            starts.append(final)
        return starts

    @torch.no_grad()
    def encode_path(self, path: str | Path, device: torch.device | str) -> Tensor:
        """Return [1,384,128,128] CPU/GPU features merged in original DOP coordinates."""
        device = torch.device(device)
        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
        _, height, width = image.shape
        tile = self.tile_size
        if height < tile or width < tile:
            pad_h, pad_w = max(0, tile - height), max(0, tile - width)
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
            height, width = image.shape[-2:]

        origins = [(y, x) for y in self._starts(height, tile, self.tile_stride)
                   for x in self._starts(width, tile, self.tile_stride)]
        crops = torch.stack([image[:, y:y + tile, x:x + tile] for y, x in origins])
        feature_sum = torch.zeros(1, self.embed_dim, self.output_grid, self.output_grid,
                                  device=device, dtype=torch.float32)
        weight_sum = torch.zeros(1, 1, self.output_grid, self.output_grid,
                                 device=device, dtype=torch.float32)

        global_x = (torch.arange(self.output_grid, device=device, dtype=torch.float32) + 0.5) * (
            width / self.output_grid)
        global_y = (torch.arange(self.output_grid, device=device, dtype=torch.float32) + 0.5) * (
            height / self.output_grid)
        for start in range(0, len(origins), self.tile_batch_size):
            crop_batch = crops[start:start + self.tile_batch_size].to(device, non_blocking=True)
            crop_batch = (crop_batch - self.image_mean) / self.image_std
            encoded = self.backbone(crop_batch, is_training=True)["x_norm_patchtokens"]
            maps = encoded.transpose(1, 2).reshape(-1, self.embed_dim, 37, 37).float()
            for offset, (y0, x0) in enumerate(origins[start:start + len(maps)]):
                x_mask = (global_x >= x0) & (global_x < x0 + tile)
                y_mask = (global_y >= y0) & (global_y < y0 + tile)
                xs = torch.where(x_mask)[0]
                ys = torch.where(y_mask)[0]
                px = global_x[xs] - x0
                py = global_y[ys] - y0
                gx = 2.0 * px / tile - 1.0
                gy = 2.0 * py / tile - 1.0
                grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
                sampling_grid = torch.stack((grid_x, grid_y), dim=-1)[None]
                sampled = F.grid_sample(maps[offset:offset + 1], sampling_grid,
                                        mode="bilinear", padding_mode="border",
                                        align_corners=False)
                wx = (torch.minimum(px, tile - px) / (tile * 0.5)).clamp_min(0.1)
                wy = (torch.minimum(py, tile - py) / (tile * 0.5)).clamp_min(0.1)
                blend = (wy[:, None] * wx[None, :])[None, None]
                feature_sum[:, :, ys[:, None], xs[None, :]] += sampled * blend
                weight_sum[:, :, ys[:, None], xs[None, :]] += blend

        return feature_sum / weight_sum.clamp_min(1e-6)

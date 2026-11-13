"""SGGA: Satellite-Guided Global Attention over 16x16 region tokens."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SGGA(nn.Module):
    """Prepare 16x16 satellite Region Tokens for joint LoGeR Global Attention."""

    region_grid = 16
    sgga_decoder_indices = (1, 3, 5, 7, 9)

    def __init__(self, in_dim: int = 384, region_dim: int = 1024,
                 pe_bands: int = 8):
        super().__init__()
        self.in_dim = int(in_dim)
        self.region_dim = int(region_dim)
        self.pe_bands = int(pe_bands)

        self.region_projection = nn.Linear(in_dim, region_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, region_dim))
        self.region_position = nn.Sequential(
            nn.Linear(4 * pe_bands, 256), nn.GELU(), nn.Linear(256, region_dim)
        )
        nn.init.zeros_(self.region_position[-1].weight)
        nn.init.zeros_(self.region_position[-1].bias)
        self.sgga_gates = nn.Parameter(torch.full((5,), 1e-3))

        self.register_buffer(
            "frequencies",
            (2.0 ** torch.arange(pe_bands, dtype=torch.float32)) * math.pi,
            persistent=False,
        )

    def initialize(self, dense_features: Tensor) -> Tensor:
        """Pool static 128x128 DINO features into the 16x16 SGGA token grid."""
        if dense_features.ndim != 4 or dense_features.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected satellite features [B,{self.in_dim},H,W], "
                f"got {tuple(dense_features.shape)}"
            )
        if tuple(dense_features.shape[-2:]) != (128, 128):
            raise ValueError("Satellite DINO dense feature map must be 128x128")
        pooled = F.adaptive_avg_pool2d(dense_features, (16, 16))
        pooled = pooled.flatten(2).transpose(1, 2)
        regions = self.region_projection(pooled)

        y = (torch.arange(16, device=dense_features.device, dtype=dense_features.dtype) + .5) / 16
        x = (torch.arange(16, device=dense_features.device, dtype=dense_features.dtype) + .5) / 16
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(1, 256, 2)
        angles = coords[..., None] * self.frequencies.to(dense_features.dtype)
        pe = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        regions = regions + self.region_position(pe) + self.modality_embedding

        return regions

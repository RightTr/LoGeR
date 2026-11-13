"""RegionViT-style Satellite Region-to-Local feature exchange."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RegionLocalBlock(nn.Module):
    """Independent self-attention groups, one region token plus its local cells."""

    def __init__(self, dim: int = 256, heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        x = tokens + self.attn(self.norm1(tokens), self.norm1(tokens),
                               self.norm1(tokens), need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class SatelliteRegionPyramid(nn.Module):
    """Build 16x16 global regions and optionally refine them with local DINO cells.

    ``dense_features`` must be a frozen DINO feature map reconstructed at 128x128
    from overlapping crops of the original DOP. Local grids are sampled from that
    map directly; no local feature is synthesized from the 16x16 region tokens.
    """

    region_grid = 16
    local_grids = (32, 48, 64, 96, 128)
    global_decoder_indices = (1, 3, 5, 7, 9)

    def __init__(self, in_dim: int = 384, region_dim: int = 1024,
                 local_dim: int = 256, heads: int = 4,
                 use_region_to_local: bool = True, pe_bands: int = 8):
        super().__init__()
        self.in_dim = int(in_dim)
        self.region_dim = int(region_dim)
        self.local_dim = int(local_dim)
        self.use_region_to_local = bool(use_region_to_local)
        self.pe_bands = int(pe_bands)

        self.region_projection = nn.Linear(in_dim, region_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, region_dim))
        self.region_position = nn.Sequential(
            nn.Linear(4 * pe_bands, 256), nn.GELU(), nn.Linear(256, region_dim)
        )
        nn.init.zeros_(self.region_position[-1].weight)
        nn.init.zeros_(self.region_position[-1].bias)
        self.joint_gates = nn.Parameter(torch.full((5,), 1e-3))

        if self.use_region_to_local:
            self.local_projection = nn.Conv2d(in_dim, local_dim, kernel_size=1)
            self.region_down = nn.Linear(region_dim, local_dim)
            self.stages = nn.ModuleList([
                RegionLocalBlock(local_dim, heads) for _ in self.local_grids
            ])
            self.region_up = nn.ModuleList([
                nn.Linear(local_dim, region_dim) for _ in self.local_grids
            ])
            self.local_up = nn.ModuleList([
                nn.Conv2d(local_dim, local_dim, kernel_size=1)
                for _ in self.local_grids
            ])
            self.region_residual_scale = nn.Parameter(torch.full((5,), 0.01))
            self.local_residual_scale = nn.Parameter(torch.full((5,), 0.01))
        else:
            self.local_projection = None
            self.region_down = None
            self.stages = None
            self.region_up = None
            self.local_up = None
            self.register_parameter("region_residual_scale", None)
            self.register_parameter("local_residual_scale", None)

        self.register_buffer(
            "frequencies",
            (2.0 ** torch.arange(pe_bands, dtype=torch.float32)) * math.pi,
            persistent=False,
        )

    def initialize(self, dense_features: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Initialize region tokens and local grids from static DINO feature maps."""
        if dense_features.ndim != 4 or dense_features.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected satellite features [B,{self.in_dim},H,W], "
                f"got {tuple(dense_features.shape)}"
            )
        if tuple(dense_features.shape[-2:]) != (128, 128):
            raise ValueError("Satellite DINO dense feature map must be 128x128")
        batch = dense_features.shape[0]
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

        if not self.use_region_to_local:
            return regions, ()
        local_pyramid = tuple(
            F.interpolate(dense_features, size=(grid, grid), mode="bilinear",
                          align_corners=False)
            for grid in self.local_grids
        )
        return regions, local_pyramid

    @staticmethod
    def _to_region_groups(local_map: Tensor, group_grid: int) -> Tensor:
        batch, channels, height, width = local_map.shape
        if height != width or height % group_grid:
            raise ValueError(f"Local grid {height} cannot be partitioned into {group_grid} regions")
        cell = height // group_grid
        return (local_map.reshape(batch, channels, group_grid, cell, group_grid, cell)
                .permute(0, 2, 4, 3, 5, 1)
                .reshape(batch * group_grid * group_grid, cell * cell, channels))

    @staticmethod
    def _from_region_groups(groups: Tensor, batch: int, channels: int,
                            grid: int, group_grid: int = 16) -> Tensor:
        cell = grid // group_grid
        return (groups.reshape(batch, group_grid, group_grid, cell, cell, channels)
                .permute(0, 5, 1, 3, 2, 4)
                .reshape(batch, channels, grid, grid))

    def refine(self, region_tokens: Tensor, local_features: Tensor,
               stage: int, previous_local: Tensor | None = None
               ) -> tuple[Tensor, Tensor | None]:
        """Exchange information within each 16x16 Region and its local cells."""
        if not self.use_region_to_local:
            return region_tokens, None
        if not 0 <= stage < len(self.local_grids):
            raise IndexError(f"Invalid Region-to-Local stage {stage}")
        grid = self.local_grids[stage]
        batch = region_tokens.shape[0]
        local_map = self.local_projection(local_features)
        if previous_local is not None:
            propagated = F.interpolate(previous_local, size=(grid, grid), mode="bilinear",
                                       align_corners=False)
            local_map = local_map + self.local_up[stage](propagated)

        region_group = self.region_down(region_tokens).reshape(batch * 256, 1, self.local_dim)
        local_group = self._to_region_groups(local_map, self.region_grid)
        grouped = self.stages[stage](torch.cat((region_group, local_group), dim=1))
        refined_region = grouped[:, 0].reshape(batch, 256, self.local_dim)
        refined_local = grouped[:, 1:]

        delta_region = self.region_up[stage](refined_region - region_group[:, 0].reshape(
            batch, 256, self.local_dim))
        region_tokens = region_tokens + self.region_residual_scale[stage] * delta_region
        local_delta = self._from_region_groups(refined_local, batch, self.local_dim, grid)
        local_map = local_map + self.local_residual_scale[stage] * local_delta
        return region_tokens, local_map

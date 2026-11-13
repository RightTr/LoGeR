"""Fixed-grid Region-to-Local Attention for SGGA satellite features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _RegionLocalBlock(nn.Module):
    """Self-attention over one Region token and its 8x8 Local tokens."""

    def __init__(self, dim: int = 256, heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        tokens = tokens + self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return tokens + self.ffn(self.norm2(tokens))


class R2LA(nn.Module):
    """Refine SGGA Region tokens using fixed, non-resampled DINO Local tokens.

    The static satellite feature map is [B,384,128,128]. It is partitioned into
    16x16 geographic Regions, each containing the corresponding 8x8=64 cells.
    Region tokens attend jointly with those cells. Each stage reads the same
    frozen Local features; no pyramid interpolation or Local hidden propagation
    is performed.
    """

    region_grid = 16
    local_grid = 128
    cells_per_region = 8
    local_tokens_per_region = cells_per_region ** 2
    decoder_indices = (1, 3, 5, 7)

    def __init__(self, in_dim: int = 384, region_dim: int = 1024,
                 dim: int = 256, heads: int = 4, pe_bands: int = 8):
        super().__init__()
        self.in_dim = int(in_dim)
        self.region_dim = int(region_dim)
        self.dim = int(dim)
        self.local_projection = nn.Conv2d(in_dim, dim, kernel_size=1)
        self.local_position = nn.Sequential(
            nn.Linear(4 * pe_bands, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.local_position[-1].weight)
        nn.init.zeros_(self.local_position[-1].bias)
        axis = (torch.arange(self.cells_per_region, dtype=torch.float32) + 0.5) / self.cells_per_region
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(1, self.local_tokens_per_region, 2)
        frequencies = (2.0 ** torch.arange(pe_bands, dtype=torch.float32)) * torch.pi
        angles = coords[..., None] * frequencies
        local_pe = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        self.register_buffer("local_pe", local_pe, persistent=False)
        self.region_down = nn.Linear(region_dim, dim)
        stages = len(self.decoder_indices)
        self.blocks = nn.ModuleList([_RegionLocalBlock(dim, heads) for _ in range(stages)])
        self.region_up = nn.ModuleList([nn.Linear(dim, region_dim) for _ in range(stages)])
        self.region_residual_scale = nn.Parameter(torch.full((stages,), 0.01))

    def prepare_local_tokens(self, dense_features: Tensor) -> Tensor:
        """Project and partition the static 128x128 feature map once per forward."""
        batch = dense_features.shape[0]
        local = self.local_projection(dense_features)
        # [B,C,16,8,16,8] -> [B*256,64,C]
        local = (local.reshape(batch, self.dim, self.region_grid, self.cells_per_region,
                               self.region_grid, self.cells_per_region)
                 .permute(0, 2, 4, 3, 5, 1)
                 .reshape(batch * self.region_grid * self.region_grid,
                          self.local_tokens_per_region, self.dim))
        return local + self.local_position(self.local_pe.to(local.dtype))

    def forward(self, region_tokens: Tensor, local_tokens: Tensor,
                stage: int) -> Tensor:
        """Update only Region tokens from fixed local cells at one R2LA stage."""
        batch = region_tokens.shape[0]
        expected_regions = self.region_grid ** 2
        region = self.region_down(region_tokens).reshape(batch * expected_regions, 1, self.dim)
        grouped = self.blocks[stage](torch.cat((region, local_tokens), dim=1))
        updated_region = grouped[:, 0].reshape(batch, expected_regions, self.dim)
        delta = self.region_up[stage](updated_region - region[:, 0].reshape(
            batch, expected_regions, self.dim))
        return region_tokens + self.region_residual_scale[stage] * delta

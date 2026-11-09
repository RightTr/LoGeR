"""RoMa-guided bidirectional UAV and satellite feature interaction."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def fourier_position(tile_bbox, grid_size=16):
    """Encode anchor centers using normalized coordinates of the full DOP."""
    batch = tile_bbox.shape[0]
    centers = (torch.arange(grid_size, device=tile_bbox.device, dtype=tile_bbox.dtype) + 0.5) / grid_size
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    local = torch.stack((xx, yy), dim=-1).reshape(1, grid_size * grid_size, 2)
    xy = tile_bbox[:, None, :2] + local * (tile_bbox[:, None, 2:] - tile_bbox[:, None, :2])
    frequencies = (2.0 ** (torch.arange(32, device=xy.device, dtype=xy.dtype) / 4)) * math.pi
    phase = xy[..., None] * frequencies
    return torch.cat((phase.sin(), phase.cos()), dim=-1).reshape(batch, grid_size * grid_size, 128)


class PriorCrossAttention(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.heads = heads
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.prior_scale = nn.Parameter(torch.tensor(0.54132485))  # softplus = 1

    def forward(self, query, source, log_prior):
        batch, length, dim = query.shape
        source_length = source.shape[1]
        q = self.q(self.q_norm(query)).reshape(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        kv = self.kv_norm(source)
        k = self.k(kv).reshape(batch, source_length, self.heads, dim // self.heads).transpose(1, 2)
        v = self.v(kv).reshape(batch, source_length, self.heads, dim // self.heads).transpose(1, 2)
        bias = (F.softplus(self.prior_scale) * log_prior[:, None]).to(q.dtype)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0)
        return self.out(result.transpose(1, 2).reshape(batch, length, dim))


class BidirectionalBlock(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.sat_attn = PriorCrossAttention(dim)
        self.uav_attn = PriorCrossAttention(dim)
        self.sat_norm = nn.LayerNorm(dim)
        self.uav_norm = nn.LayerNorm(dim)
        self.sat_ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.uav_ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, uav, satellite, prior, reliability):
        mixed = 0.99 * prior + 0.01 / prior.shape[-1]
        reverse = mixed * reliability[..., None] + 1e-6
        satellite = satellite + self.sat_attn(satellite, uav, reverse.transpose(1, 2).log())
        satellite = satellite + self.sat_ffn(self.sat_norm(satellite))
        gate = reliability[..., None]
        uav = uav + gate * self.uav_attn(uav, satellite, mixed.log())
        uav = uav + gate * self.uav_ffn(self.uav_norm(uav))
        return uav, satellite


class SatelliteBidirectional(nn.Module):
    def __init__(self, hidden_dim=1024, feature_dim=1024, dim=256, layers=4):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_dim, dim)
        self.uav_proj = nn.Linear(feature_dim, dim)
        self.sat_proj = nn.Linear(feature_dim, dim)
        self.pos_proj = nn.Linear(128, dim)
        self.memory_proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(BidirectionalBlock(dim) for _ in range(layers))
        self.output = nn.Linear(dim, hidden_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden, uav_features, satellite_features, tile_bbox, previous_memory, prior, reliability):
        batch, frames, patches, channels = hidden.shape
        uav = self.hidden_proj(hidden.reshape(batch, frames * patches, channels))
        uav = uav + self.uav_proj(uav_features.reshape(batch, frames * patches, -1))
        satellite = self.sat_proj(satellite_features)
        satellite = satellite + self.pos_proj(fourier_position(tile_bbox))
        satellite = satellite + self.memory_proj(previous_memory)
        for block in self.blocks:
            uav, satellite = block(uav, satellite, prior, reliability)
        return self.output(uav).reshape(batch, frames, patches, channels), satellite

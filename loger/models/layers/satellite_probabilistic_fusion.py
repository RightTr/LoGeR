"""RoMa-probability-routed UAV and satellite feature exchange."""

import math
import torch
from torch import nn


def fourier_position(tile_bbox, grid_size=16):
    """Fourier encode satellite anchor centers in normalized map coordinates."""
    batch = tile_bbox.shape[0]
    centers = (torch.arange(grid_size, device=tile_bbox.device,
                            dtype=tile_bbox.dtype) + 0.5) / grid_size
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    local = torch.stack((xx, yy), dim=-1).reshape(1, grid_size * grid_size, 2)
    xy = tile_bbox[:, None, :2] + local * (
        tile_bbox[:, None, 2:] - tile_bbox[:, None, :2])
    frequencies = (2.0 ** (torch.arange(32, device=xy.device, dtype=xy.dtype) / 4)) * math.pi
    phase = xy[..., None] * frequencies
    return torch.cat((phase.sin(), phase.cos()), dim=-1).reshape(
        batch, grid_size * grid_size, 128)


class SatelliteProbabilisticFusion(nn.Module):
    """Exchange features through RoMa's fixed spatial association distribution.

    The association is conditional over the 16x16 satellite grid. Match
    probability and RoMa certainty remain separate gates so no-match mass is
    never silently renormalized into a confident spatial correspondence.
    """

    def __init__(self, hidden_dim=1024, feature_dim=1024, dim=256,
                 grid_size=16, topk=8):
        super().__init__()
        self.grid_size = int(grid_size)
        self.anchors = self.grid_size * self.grid_size
        self.topk = min(int(topk), self.anchors)
        self.hidden_proj = nn.Linear(hidden_dim, dim)
        self.satellite_proj = nn.Linear(feature_dim, dim)
        self.position_proj = nn.Linear(128, dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.write_delta = nn.Sequential(
            nn.LayerNorm(2 * dim + 1),
            nn.Linear(2 * dim + 1, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.write_gate = nn.Linear(2 * dim + 1, dim)
        self.read_out = nn.Linear(3 * dim, hidden_dim)
        nn.init.constant_(self.write_gate.bias, -2.0)
        nn.init.zeros_(self.read_out.weight)
        nn.init.zeros_(self.read_out.bias)

    def forward(self, hidden, satellite_features, tile_bbox, previous_memory,
                prior, reliability, match_probability, frame_mask=None,
                use_memory=False):
        batch, frames, patches, channels = hidden.shape
        query_count = frames * patches
        prior = prior.float().reshape(batch, query_count, self.anchors)
        reliability = reliability.float().reshape(batch, query_count).clamp(0, 1)
        match_probability = match_probability.float().reshape(batch, query_count).clamp(0, 1)
        route_gate = reliability * match_probability

        position = fourier_position(tile_bbox, self.grid_size)
        if previous_memory is None or not use_memory:
            # Bind each DINO anchor to its map coordinate once at initialization.
            memory = self.satellite_proj(satellite_features) + self.position_proj(position)
        else:
            memory = previous_memory
        satellite = self.memory_norm(memory)

        # Full conditional distribution is used for reading. Confidence gates
        # its mass, so an unmatched query contributes zero rather than a fake
        # normalized match.
        read_weights = prior * route_gate[..., None]
        read_static = torch.bmm(read_weights.to(satellite.dtype), satellite)

        # Sparse Top-k write. Keep the original probability mass after pruning.
        top_values, top_indices = prior.topk(self.topk, dim=-1)
        write_weights = top_values * route_gate[..., None]
        if frame_mask is not None:
            frame_mask = frame_mask.to(write_weights.device, write_weights.dtype)
            frame_mask = frame_mask[:, :, None].expand(-1, -1, patches).reshape(batch, query_count)
            write_weights = write_weights * frame_mask[..., None]

        values = self.hidden_proj(hidden.reshape(batch, query_count, channels))
        weighted_values = values.float()[:, :, None, :] * write_weights[..., None]
        weighted_values = weighted_values.reshape(batch, query_count * self.topk, -1)
        indices = top_indices.reshape(batch, query_count * self.topk)
        write_sum = torch.zeros(batch, self.anchors, values.shape[-1],
                                device=values.device, dtype=torch.float32)
        write_sum.scatter_add_(1, indices[..., None].expand_as(weighted_values), weighted_values)
        coverage = write_weights.new_zeros(batch, self.anchors)
        coverage.scatter_add_(1, indices, write_weights.reshape(batch, -1))
        observation = (write_sum / coverage[..., None].clamp_min(1e-8)).to(memory.dtype)

        write_input = torch.cat((memory, observation,
                                 coverage[..., None].to(memory.dtype)), dim=-1)
        update = self.write_delta(write_input)
        gate = torch.sigmoid(self.write_gate(write_input))
        updated_memory = memory + gate * update * (coverage > 0)[..., None].to(update.dtype)
        satellite_updated = self.memory_norm(updated_memory)
        read_multi = torch.bmm(read_weights.to(satellite_updated.dtype), satellite_updated)
        delta = self.read_out(torch.cat((values, read_static, read_multi), dim=-1))
        delta = delta.reshape(batch, frames, patches, channels)

        stats = {
            "coverage": (coverage > 0).float().mean(),
            "write_mass": coverage.sum(dim=-1).mean(),
            "retained_topk_mass": top_values.sum(dim=-1).mean(),
            "delta_ratio": delta.float().norm() / hidden.float().norm().clamp_min(1e-8),
            "memory_update_norm": (updated_memory - memory).float().norm(dim=-1).mean(),
        }
        return delta, updated_memory, coverage, stats

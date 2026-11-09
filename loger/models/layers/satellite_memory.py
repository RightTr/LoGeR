"""Persistent dynamic state for one fixed satellite tile."""

import torch
from torch import nn


class SatelliteMemory(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.candidate = nn.Linear(dim, dim)
        self.gate = nn.Linear(2 * dim + 1, dim)

    def forward(self, previous, satellite, prior, reliability, frame_mask):
        batch, frames = frame_mask.shape
        patches = reliability.shape[1] // frames
        valid = (reliability.reshape(batch, frames, patches) >= 0.5) & frame_mask[:, :, None]
        top_values, top_indices = prior.topk(4, dim=-1)
        coverage = prior.new_zeros(batch, 256)
        coverage.scatter_add_(1, top_indices.reshape(batch, -1),
                              (top_values * reliability[..., None] * valid.reshape(batch, -1, 1)).reshape(batch, -1))
        candidate = self.candidate(satellite)
        gate = torch.sigmoid(self.gate(torch.cat((previous, candidate,
                                                  coverage[..., None].to(candidate.dtype)), dim=-1)))
        updated = previous + gate * (candidate - previous)
        return torch.where(coverage[..., None] > 0, updated, previous)

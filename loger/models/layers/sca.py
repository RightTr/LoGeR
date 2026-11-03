"""Spatial UAV queries over a shared satellite memory."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCA(nn.Module):
    def __init__(self, uav_dim, satellite_dim=384, attn_dim=256):
        super().__init__()
        self.query_norm = nn.LayerNorm(uav_dim)
        self.query = nn.Linear(uav_dim, attn_dim, bias=False)
        self.satellite_norm = nn.LayerNorm(satellite_dim)
        self.key = nn.Linear(satellite_dim, attn_dim, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(0.07).log())

    def forward(self, hidden, batch, frames, patch_start, grid, memory):
        height, width = grid
        patches = hidden[:, patch_start:].transpose(1, 2).reshape(batch * frames, -1, height, width)
        pooled = F.adaptive_avg_pool2d(patches, 4).flatten(2).transpose(1, 2)
        queries = F.normalize(self.query(self.query_norm(pooled)).float(), dim=-1)
        tokens = memory["tokens"]
        keys = F.normalize(self.key(self.satellite_norm(tokens)).float(), dim=-1)
        logits = queries @ keys.transpose(-1, -2) / self.log_temperature.exp().clamp_min(1e-4)
        logits = logits.reshape(batch, frames, 16, -1)
        logits = logits.masked_fill(~memory["mask"][:, None, None, :], torch.finfo(logits.dtype).min)
        spatial = logits.softmax(dim=-1)
        frame = spatial.mean(dim=2)
        return {"spatial": spatial, "frame": frame, "chunk": frame.mean(dim=1)}

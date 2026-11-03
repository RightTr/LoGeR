"""Satellite context for the UAV patch grid before SWA."""

import math
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
        self.value = nn.Linear(satellite_dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, uav_dim)
        self.gate = nn.Parameter(torch.tensor(0.01))
        self.register_buffer("frequencies", torch.logspace(0, 2, satellite_dim // 4), persistent=False)

    def forward(self, hidden, batch, frames, patch_start, grid, memory):
        height, width = grid
        patches = hidden[:, patch_start:].transpose(1, 2).reshape(batch * frames, -1, height, width)
        pooled = F.adaptive_avg_pool2d(patches, 4).flatten(2).transpose(1, 2)
        chunk = pooled.reshape(batch, frames * 16, -1).mean(dim=1, keepdim=True)
        queries = self.query(self.query_norm(pooled.reshape(batch, frames, 16, -1) + chunk[:, None]))
        coords = memory["coords"].float() / memory["dop_size"].float()
        angles = coords[..., None] * self.frequencies * (2 * math.pi)
        position = torch.cat((angles[..., 0, :].sin(), angles[..., 0, :].cos(),
                              angles[..., 1, :].sin(), angles[..., 1, :].cos()), dim=-1)
        tokens = self.satellite_norm(memory["tokens"] + position.to(memory["tokens"].dtype))
        keys = self.key(tokens)
        values = self.value(tokens)
        logits = queries.float() @ keys.float().transpose(-1, -2) / math.sqrt(keys.shape[-1])
        logits = logits.reshape(batch, frames, 16, -1)
        logits = logits.masked_fill(~memory["mask"][:, None, None, :], torch.finfo(logits.dtype).min)
        spatial = logits.softmax(dim=-1)
        context = spatial @ values.float()
        context = context.reshape(batch * frames, 4, 4, -1).permute(0, 3, 1, 2)
        context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=False)
        delta = self.output(context.permute(0, 2, 3, 1).reshape(batch * frames, height * width, -1).to(hidden.dtype))
        enhanced = torch.cat((hidden[:, :patch_start], hidden[:, patch_start:] + self.gate * delta), dim=1)
        frame = spatial.mean(dim=2)
        return enhanced, {"frame": frame, "chunk": frame.mean(dim=1)}

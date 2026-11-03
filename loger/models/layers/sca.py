"""Shared satellite cross-attention before the first SWA layer."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCA(nn.Module):
    def __init__(self, uav_dim, satellite_dim, attn_dim=256, num_heads=8, gate_init=0.01):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.query_norm = nn.LayerNorm(uav_dim)
        self.satellite_norm = nn.LayerNorm(satellite_dim)
        self.query = nn.Linear(uav_dim, attn_dim, bias=False)
        self.key = nn.Linear(satellite_dim, attn_dim, bias=False)
        self.value = nn.Linear(satellite_dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, uav_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.register_buffer("frequencies", torch.logspace(0, 2, satellite_dim // 4), persistent=False)

    def prepare_satellite(self, tokens, coords, valid_mask):
        coords = coords.float()
        valid_mask = valid_mask.bool()
        minimum = coords.masked_fill(~valid_mask[..., None], torch.inf).amin(dim=1, keepdim=True)
        maximum = coords.masked_fill(~valid_mask[..., None], -torch.inf).amax(dim=1, keepdim=True)
        xy = 2 * (coords - minimum) / (maximum - minimum).clamp_min(1) - 1
        angles = xy[..., None] * self.frequencies * math.pi
        position = torch.cat((angles[..., 0, :].sin(), angles[..., 0, :].cos(),
                              angles[..., 1, :].sin(), angles[..., 1, :].cos()), dim=-1)
        satellite = self.satellite_norm(tokens + position.to(tokens.dtype))
        batch = satellite.shape[0]
        keys = self.key(satellite).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(satellite).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return keys, values, valid_mask

    def forward(self, hidden, batch_size, num_frames, patch_start_idx, satellite_memory):
        patches = hidden[:, patch_start_idx:]
        frame_features = patches.reshape(batch_size, num_frames, -1, hidden.shape[-1]).mean(dim=2)
        queries = self.query(self.query_norm(frame_features))
        queries = queries.reshape(batch_size, num_frames, self.num_heads, self.head_dim).transpose(1, 2)
        keys, values, mask = satellite_memory
        if keys.shape[0] == 1:
            keys = keys.expand(batch_size, -1, -1, -1)
            values = values.expand(batch_size, -1, -1, -1)
            mask = mask.expand(batch_size, -1)
        context = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask[:, None, None, :])
        context = context.transpose(1, 2).reshape(batch_size * num_frames, -1)
        delta = self.output(context).unsqueeze(1)
        return torch.cat((hidden[:, :patch_start_idx], patches + self.gate * delta), dim=1)

"""Patch-aligned residual adapter for RoMa GP features."""

import torch
from torch import nn


class RoMaGeometryAdapter(nn.Module):
    def __init__(self, dim=1024, hidden_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        self.gate = nn.Parameter(torch.tensor(0.01))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden, roma_tokens, patch_start):
        if hidden[:, patch_start:].shape != roma_tokens.shape:
            raise ValueError("RoMa tokens must match the UAV patch tokens")
        delta = self.mlp(self.norm(roma_tokens.to(dtype=self.norm.weight.dtype)))
        patches = hidden[:, patch_start:] + (self.gate * delta).to(hidden.dtype)
        return torch.cat((hidden[:, :patch_start], patches), dim=1)

"""Small residual adapter from frozen RoMa tokens into frozen LoGeR."""

import torch
from torch import nn


class RoMaLoGeRFusion(nn.Module):
    def __init__(self, dim=1024, hidden_dim=256):
        super().__init__()
        self.loger_norm = nn.LayerNorm(dim)
        self.roma_norm = nn.LayerNorm(dim)
        self.fusion = nn.Linear(2 * dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, loger_hidden, roma_hidden):
        if loger_hidden.shape != roma_hidden.shape:
            raise ValueError(
                f"RoMa and LoGeR patch grids/features must match; got "
                f"{tuple(roma_hidden.shape)} vs {tuple(loger_hidden.shape)}")
        features = torch.cat((self.loger_norm(loger_hidden),
                              self.roma_norm(roma_hidden)), dim=-1)
        delta = self.output(torch.nn.functional.gelu(self.fusion(features)))
        stats = {
            "delta_ratio": delta.float().norm() /
                           loger_hidden.float().norm().clamp_min(1e-8),
            "roma_hidden_norm": roma_hidden.float().norm(dim=-1).mean(),
        }
        return delta, stats

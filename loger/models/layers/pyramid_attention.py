"""Independent satellite attention at three spatial scales."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidAttention(nn.Module):
    def __init__(self, uav_dim=1024, satellite_dim=384, attention_dim=256):
        super().__init__()
        self.query_norm = nn.LayerNorm(uav_dim)
        self.query = nn.Linear(uav_dim, attention_dim, bias=False)
        self.key_norm = nn.ModuleDict({name: nn.LayerNorm(satellite_dim) for name in ("L0", "L1", "L2")})
        self.keys = nn.ModuleDict({name: nn.Linear(satellite_dim, attention_dim, bias=False)
                                   for name in ("L0", "L1", "L2")})
        self.log_temperature = nn.ParameterDict({name: nn.Parameter(torch.tensor(0.07).log())
                                                  for name in ("L0", "L1", "L2")})

    def forward(self, hidden, batch_size, frames, patch_start_idx, memory):
        patches = hidden[:, patch_start_idx:].reshape(batch_size, frames, -1, hidden.shape[-1])
        query = F.normalize(self.query(self.query_norm(patches.mean(dim=2))).float(), dim=-1)
        output = {}
        for name, level in memory.items():
            tokens = level["tokens"]
            key = F.normalize(self.keys[name](self.key_norm[name](tokens)).float(), dim=-1)
            logits = query @ key.transpose(-1, -2) / self.log_temperature[name].exp().clamp_min(1e-4)
            logits = logits.masked_fill(~level["mask"][:, None], torch.finfo(logits.dtype).min)
            output[name] = logits.softmax(dim=-1)
        return output

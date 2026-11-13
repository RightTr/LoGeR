"""MatchAnything RoMa scale-16 Transformer classification decoder."""

import torch
from torch import nn

from .layers.block import Block
from .layers.attention import MemEffAttention


class TransformerDecoder(nn.Module):
    """Five MatchAnything RoMa blocks and the 64x64-anchor classification head."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(1024, 8, attn_class=MemEffAttention) for _ in range(5)
        ])
        self.to_out = nn.Linear(1024, 64 * 64 + 1)

    def forward(self, gp_posterior, features):
        batch, _, height, width = gp_posterior.shape
        with torch.autocast(device_type=gp_posterior.device.type, enabled=False):
            tokens = torch.cat((gp_posterior, features), dim=1).flatten(2).transpose(1, 2)
            hidden = self.blocks(tokens.float())
            output = self.to_out(hidden)
        output = output.transpose(1, 2).reshape(batch, 4097, height, width)
        return output[:, :4096], output[:, 4096:]

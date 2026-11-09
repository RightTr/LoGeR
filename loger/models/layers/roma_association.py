"""Convert MatchAnything RoMa's 64x64 anchors to 16x16 satellite memory."""

import torch.nn.functional as F
from torch import nn


class RoMaAssociation(nn.Module):
    def forward(self, match_logits, certainty, roma_grid, uav_grid):
        batch, frames, _, _ = match_logits.shape
        source_h, source_w = roma_grid
        target_h, target_w = uav_grid
        probs = match_logits.float().softmax(dim=-1)
        # Each memory anchor receives the probability mass of its 4x4 region.
        probs = probs.reshape(batch * frames, source_h * source_w, 16, 4, 16, 4)
        probs = probs.sum(dim=(3, 5)).reshape(batch * frames, source_h * source_w, 256)
        certainty = certainty.float().reshape(batch * frames, 1, source_h, source_w)
        if (source_h, source_w) != (target_h, target_w):
            probs = probs.transpose(1, 2).reshape(batch * frames, 256, source_h, source_w)
            probs = F.interpolate(probs, size=(target_h, target_w), mode="bilinear", align_corners=False)
            probs = probs.flatten(2).transpose(1, 2)
            certainty = F.interpolate(certainty, size=(target_h, target_w), mode="bilinear", align_corners=False)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs.reshape(batch, frames * target_h * target_w, 256), certainty.flatten(1).reshape(batch, frames * target_h * target_w)

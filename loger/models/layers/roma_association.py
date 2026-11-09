"""Convert MatchAnything RoMa's 64x64 anchors to 16x16 satellite memory."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class RoMaAssociation(nn.Module):
    def forward(self, match_logits, no_match_logits, roma_grid, uav_grid):
        batch, frames, _, _ = match_logits.shape
        source_h, source_w = roma_grid
        target_h, target_w = uav_grid
        spatial_probs = match_logits.float().softmax(dim=-1)
        no_match_logits = no_match_logits.float().reshape(
            batch * frames, source_h * source_w, 1)
        match_mass = torch.cat((match_logits.reshape(batch * frames, source_h * source_w, -1).float(),
                                no_match_logits), dim=-1).softmax(dim=-1)[..., :-1].sum(dim=-1)
        legacy_reliability = no_match_logits.sigmoid().reshape(
            batch * frames, 1, source_h, source_w)
        # Each memory anchor receives the probability mass of its 4x4 region.
        probs = spatial_probs.reshape(batch * frames, source_h * source_w, 16, 4, 16, 4)
        probs = probs.sum(dim=(3, 5)).reshape(batch * frames, source_h * source_w, 256)
        if (source_h, source_w) != (target_h, target_w):
            probs = probs.transpose(1, 2).reshape(batch * frames, 256, source_h, source_w)
            probs = F.interpolate(probs, size=(target_h, target_w), mode="bilinear", align_corners=False)
            probs = probs.flatten(2).transpose(1, 2)
            match_mass = F.interpolate(
                match_mass.reshape(batch * frames, 1, source_h, source_w),
                size=(target_h, target_w), mode="bilinear", align_corners=False).flatten(1)
            legacy_reliability = F.interpolate(
                legacy_reliability, size=(target_h, target_w), mode="bilinear",
                align_corners=False)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # Entropy measures how concentrated the conditional spatial match is;
        # match_mass separately preserves the classifier's no-match probability.
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
        certainty = (1.0 - entropy / math.log(256)).clamp(0, 1)
        return (probs.reshape(batch, frames * target_h * target_w, 256),
                legacy_reliability.flatten(1).reshape(batch, frames * target_h * target_w),
                match_mass.reshape(batch, frames * target_h * target_w).clamp(0, 1),
                certainty.reshape(batch, frames * target_h * target_w))

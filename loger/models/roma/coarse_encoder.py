"""RoMa feature interaction before the Transformer decoder."""

import torch
from torch import nn

from .checkpoint import load_coarse_weights
from .matcher import CosKernel, GP


class RoMaCoarseEncoder(nn.Module):
    """Project shared DINO features and compute satellite-conditioned tokens."""

    def __init__(self, roma_checkpoint=None):
        super().__init__()
        self.proj = nn.ModuleDict({"16": nn.Sequential(
            nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512))})
        self.satellite_proj = nn.ModuleDict({"16": nn.Sequential(
            nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512))})
        self.gps = nn.ModuleDict({"16": GP(
            CosKernel, T=0.2, learn_temperature=False, gp_dim=512,
            basis="fourier", no_cov=True, sigma_noise=0.1)})
        if roma_checkpoint is not None:
            load_coarse_weights(self, roma_checkpoint)
        self.gps.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.gps.eval()
        return self

    def forward_from_features(self, f1, f2, uav_grid_size, satellite_grid_size):
        f1 = f1.transpose(1, 2).reshape(f1.shape[0], 1024, *uav_grid_size)
        f2 = f2.transpose(1, 2).reshape(f2.shape[0], 1024, *satellite_grid_size)
        with torch.autocast(device_type=f1.device.type, enabled=False):
            f1_s = self.proj["16"](f1.float())
            f2_s = self.satellite_proj["16"](f2.float())
            gp_posterior = self.gps["16"](f1_s, f2_s)
        coarse_tokens = torch.cat((gp_posterior, f1_s), dim=1).flatten(2).transpose(1, 2)
        return {"f1_s": f1_s, "f2_s": f2_s, "gp_posterior": gp_posterior,
                "coarse_tokens": coarse_tokens, "grid_size": tuple(f1_s.shape[-2:])}

    forward = forward_from_features

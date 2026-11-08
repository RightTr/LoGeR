"""The DINO-only path of RoMa's CNNandDinov2 encoder."""

import torch
from torch import nn

from ..dinov2.hub.backbones import dinov2_vitl14


class CNNandDinov2(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.dinov2_vitl14 = backbone if backbone is not None else dinov2_vitl14(
            pretrained=False, img_size=518,
        )

    def forward_coarse(self, images):
        batch, _, height, width = images.shape
        parameter = next(self.dinov2_vitl14.parameters())
        with torch.no_grad():
            features = self.dinov2_vitl14.forward_features(
                images.to(device=parameter.device, dtype=parameter.dtype)
            )["x_norm_patchtokens"]
        return features.permute(0, 2, 1).reshape(batch, 1024, height // 14, width // 14)

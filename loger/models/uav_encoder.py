"""UAV image encoder based on the LoGeR DINOv2-L/14-Reg backbone."""

import torch.nn as nn

from .dinov2.hub.backbones import dinov2_vitl14_reg


class UAVEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = dinov2_vitl14_reg(pretrained=pretrained)
        del self.backbone.mask_token
        self.patch_size = 14
        self.embed_dim = self.backbone.blocks[0].attn.qkv.in_features

    def forward(self, images, is_training=True):
        return self.backbone(images, is_training=is_training)

    @staticmethod
    def remap_state_dict(state_dict):
        """Map older encoder prefixes to the current backbone prefix."""
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                key = "uav_encoder.backbone." + key[len("encoder."):]
            elif key.startswith("uav_encoder.") and not key.startswith("uav_encoder.backbone."):
                key = "uav_encoder.backbone." + key[len("uav_encoder."):]
            remapped[key] = value
        return remapped

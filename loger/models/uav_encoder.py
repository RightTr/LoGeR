"""UAV image encoder based on the LoGeR DINOv2-L/14-Reg backbone."""

from collections import OrderedDict

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
        """Map legacy and intermediate Pi3 UAV keys to this encoder's layout."""
        def remap(key):
            if key == "encoder" or key.startswith("encoder."):
                return "uav_encoder.backbone" + key[len("encoder"):]
            if key == "uav_encoder" or (
                key.startswith("uav_encoder.") and not key.startswith("uav_encoder.backbone.")
            ):
                return "uav_encoder.backbone" + key[len("uav_encoder"):]
            return key

        remapped = OrderedDict()
        for key, value in state_dict.items():
            new_key = remap(key)
            if new_key in remapped:
                raise RuntimeError(f"Checkpoint contains conflicting parameter keys for {new_key!r}")
            remapped[new_key] = value

        if hasattr(state_dict, "_metadata"):
            metadata = OrderedDict()
            for key, value in state_dict._metadata.items():
                new_key = remap(key)
                if new_key in metadata:
                    raise RuntimeError(f"Checkpoint contains conflicting module metadata for {new_key!r}")
                metadata[new_key] = value
            remapped._metadata = metadata

        return remapped

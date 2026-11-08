"""Official RoMa Outdoor coarse path for GP feature distillation."""

import torch
from torch import nn

from .matcher import CosKernel, GP
from .transformer.dinov2 import vit_large


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _load_prefix(module, weights, prefix):
    selected = {name[len(prefix):]: value for name, value in weights.items()
                if name.startswith(prefix)}
    expected = module.state_dict()
    if selected.keys() != expected.keys():
        raise RuntimeError(f"Incomplete RoMa weights for {prefix}: "
                           f"missing={sorted(expected.keys() - selected.keys())}, "
                           f"unexpected={sorted(selected.keys() - expected.keys())}")
    module.load_state_dict(selected, strict=True)


class RoMaGPTeacher(nn.Module):
    """DINOv2-L -> shared projection -> GP; no VGG or Transformer decoder."""

    def __init__(self, dino_checkpoint, roma_checkpoint):
        super().__init__()
        self.dinov2 = vit_large(img_size=518, patch_size=14, init_values=1.0,
                                ffn_layer="mlp", block_chunks=0)
        self.dinov2.load_state_dict(_load(dino_checkpoint), strict=True)
        self.proj = nn.Sequential(nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512))
        self.gp = GP(CosKernel, T=0.2, learn_temperature=False, gp_dim=512,
                     basis="fourier", no_cov=True, sigma_noise=0.1)
        weights = _load(roma_checkpoint)
        _load_prefix(self.proj, weights, "decoder.proj.16.")
        _load_prefix(self.gp, weights, "decoder.gps.16.")
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, image_a, image_b):
        def features(image):
            batch, _, height, width = image.shape
            tokens = self.dinov2.forward_features(image)["x_norm_patchtokens"]
            return tokens.transpose(1, 2).reshape(batch, 1024, height // 14, width // 14)

        with torch.autocast(device_type=image_a.device.type, enabled=False):
            f1 = self.proj(features(image_a.float()))
            f2 = self.proj(features(image_b.float()))
            return self.gp(f1, f2)

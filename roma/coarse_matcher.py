"""MatchAnything RoMa coarse GP and five-block matching interface."""

from pathlib import Path

import torch
from torch import nn

from .matcher import CosKernel, GP
from .transformer.dinov2 import vit_large
from .transformer import TransformerDecoder


def _load(path):
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _load_prefix(module, weights, prefix):
    prefix = "matcher.model." + prefix
    selected = {name[len(prefix):]: value for name, value in weights.items()
                if name.startswith(prefix)}
    expected = module.state_dict()
    if selected.keys() != expected.keys():
        raise RuntimeError(f"Incomplete RoMa weights for {prefix}: "
                           f"missing={sorted(expected.keys() - selected.keys())}, "
                           f"unexpected={sorted(selected.keys() - expected.keys())}")
    module.load_state_dict(selected, strict=True)


class RoMaCoarseMatcher(nn.Module):
    """Frozen MatchAnything RoMa coarse matcher, without VGG or refiners.

    Inputs are ImageNet-normalized RGB tensors shaped [B, 3, H, W].
    ``match_logits`` and ``match_prob`` are [B, H/14*W/14, 4096].
    """

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
        self.embedding_decoder = TransformerDecoder()
        _load_prefix(self.embedding_decoder, weights, "decoder.embedding_decoder.")
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def project_image(self, image):
        batch, _, height, width = image.shape
        with torch.autocast(device_type=image.device.type, enabled=False):
            tokens = self.dinov2.forward_features(image.float())["x_norm_patchtokens"]
            features = tokens.transpose(1, 2).reshape(batch, 1024, height // 14, width // 14)
            return self.proj(features)

    def project_features(self, image_a, image_b):
        return self.project_image(image_a), self.project_image(image_b)

    def encode_target(self, image_b):
        return self.project_image(image_b)

    @torch.no_grad()
    def forward(self, image_a, image_b):
        return self.match_with_target(image_a, self.encode_target(image_b), image_b.shape[-2:])

    @torch.no_grad()
    def match_with_target(self, image_a, target_features, target_size):
        f1_s = self.project_image(image_a)
        f2_s = target_features
        with torch.autocast(device_type=image_a.device.type, enabled=False):
            gp_posterior = self.gp(f1_s, f2_s)
            logits, certainty = self.embedding_decoder(gp_posterior, f1_s)
        batch, _, height, width = logits.shape
        logits = logits.flatten(2).transpose(1, 2)
        certainty = certainty.flatten(2).transpose(1, 2).squeeze(-1)
        return {
            "match_logits": logits,
            "match_prob": logits.softmax(dim=-1),
            "certainty_logits": certainty,
            "certainty": certainty.sigmoid(),
            "gp_posterior": gp_posterior,
            "grid_size": (height, width),
            "target_size": tuple(target_size),
        }

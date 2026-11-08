"""Encode a complete satellite image once and save its DINO patch features."""

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F


def save_satellite_features(image_path, output_path, encoder):
    """Save features in the format accepted by ``Pi3`` as satellite memory.

    The RGB image is read in full and normalized using ImageNet statistics.
    Right/bottom padding preserves all pixels when a dimension is not divisible
    by 14. The caller supplies the existing SatelliteEncoder and controls its
    device and pretrained weights.
    """
    with rasterio.open(image_path) as dataset:
        source = dataset.read(indexes=(1, 2, 3), masked=True)
        raw = np.asarray(source.filled(0))

    scale = np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype, np.integer) else 1.0
    image = np.clip(np.nan_to_num(raw.astype(np.float32)) / scale, 0.0, 1.0)
    height, width = image.shape[-2:]
    image = torch.from_numpy(image).unsqueeze(0)
    mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    image = (image - mean) / std
    image = F.pad(image, (0, (-width) % 14, 0, (-height) % 14), mode="replicate")

    with torch.no_grad():
        tokens = encoder(image)["patch_tokens"].detach().cpu()
    features = {
        "patch_tokens": tokens.unsqueeze(1),
        "grid_size": (image.shape[-2] // 14, image.shape[-1] // 14),
        "image_size": (height, width),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    return features

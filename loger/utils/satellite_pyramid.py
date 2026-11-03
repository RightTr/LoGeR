"""Three satellite views with coordinates in the original DOP."""

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
import torch.nn.functional as F

from .satellite import SatelliteTileManager


def read_pyramid(path, center):
    with rasterio.open(path) as source:
        width, height = source.width, source.height
        scaled_width = round(518 * width / max(width, height))
        scaled_height = round(518 * height / max(width, height))
        image = source.read((1, 2, 3), out_shape=(3, scaled_height, scaled_width),
                            resampling=Resampling.bilinear)
        valid = source.dataset_mask(out_shape=(scaled_height, scaled_width),
                                    resampling=Resampling.nearest) > 0
    scale = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
    image = np.clip(image.astype(np.float32) / scale, 0, 1)
    image = (image - SatelliteTileManager.mean) / SatelliteTileManager.std
    left, top = (518 - scaled_width) // 2, (518 - scaled_height) // 2
    global_tile = torch.zeros(3, 518, 518)
    global_tile[:, top:top + scaled_height, left:left + scaled_width] = torch.from_numpy(image)
    global_valid = torch.zeros(518, 518, dtype=torch.bool)
    global_valid[top:top + scaled_height, left:left + scaled_width] = torch.from_numpy(valid)

    axis = (torch.arange(37).float() + 0.5) * 14
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    global_coords = torch.stack(((xx - left) * width / scaled_width,
                                 (yy - top) * height / scaled_height), dim=-1)
    global_patch_valid = F.avg_pool2d(global_valid[None, None].float(), 14, 14)[0, 0] > 0
    global_pool_coords = F.adaptive_avg_pool2d(global_coords.permute(2, 0, 1)[None], 16)[0].permute(1, 2, 0)
    global_pool_valid = F.adaptive_max_pool2d(global_patch_valid[None, None].float(), 16)[0, 0].bool()

    with SatelliteTileManager(path) as manager:
        local = manager.get_tiles(*center)
    return {
        "global_tile": global_tile,
        "global_coords": global_pool_coords.reshape(256, 2),
        "global_valid": global_pool_valid.reshape(256),
        "local": local,
        "width": width,
        "height": height,
    }

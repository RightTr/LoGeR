"""TensorBoard overlays in original DOP coordinates."""

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling


def dop_image(path, max_width=900):
    with rasterio.open(path) as source:
        width = min(source.width, max_width)
        height = round(source.height * width / source.width)
        image = source.read((1, 2, 3), out_shape=(3, height, width), resampling=Resampling.bilinear)
        original = (source.width, source.height)
    scale = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
    rgb = np.clip(image.astype(np.float32) / scale, 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8), original


def overlay(image, original, weights, coords):
    width, height = original
    shown_h, shown_w = image.shape[:2]
    density = np.zeros((shown_h, shown_w), np.float32)
    pixel_area = 518 / 16
    for (x, y), weight in zip(coords, weights):
        x0 = max(0, round((x - pixel_area / 2) * shown_w / width))
        x1 = min(shown_w, round((x + pixel_area / 2) * shown_w / width))
        y0 = max(0, round((y - pixel_area / 2) * shown_h / height))
        y1 = min(shown_h, round((y + pixel_area / 2) * shown_h / height))
        if x1 > x0 and y1 > y0:
            density[y0:y1, x0:x1] += weight / ((x1 - x0) * (y1 - y0))
    density /= max(float(density.max()), 1e-12)
    heat = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.55,
                                        heat, 0.45, 0), cv2.COLOR_BGR2RGB)

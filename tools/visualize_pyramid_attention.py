"""Render three attention maps in original DOP pixel coordinates."""

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling


def attention_panel(dop, level, weights, gt, tile_origins=()):
    height, width = dop.shape[:2]
    density = np.zeros((height, width), np.float32)
    coords = level["coords"][0].cpu().numpy()
    valid = level["mask"][0].cpu().numpy()
    size_x, size_y = ((width / 16, height / 16) if level["name"] == "L0" else
                      (518 / 16, 518 / 16) if level["name"] == "L1" else (14, 14))
    for (x, y), weight, keep in zip(coords, weights, valid):
        if not keep:
            continue
        x0, x1 = max(0, round(x - size_x / 2)), min(width, round(x + size_x / 2))
        y0, y1 = max(0, round(y - size_y / 2)), min(height, round(y + size_y / 2))
        if x1 > x0 and y1 > y0:
            density[y0:y1, x0:x1] += weight / ((x1 - x0) * (y1 - y0))
    density /= max(density.max(), 1e-12)
    heat = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
    panel = cv2.addWeighted(dop, 0.55, heat, 0.45, 0)
    for x, y in tile_origins:
        cv2.rectangle(panel, (int(x), int(y)), (int(x + 518), int(y + 518)), (255, 255, 255), 2)
    cv2.circle(panel, tuple(map(round, gt)), 12, (0, 255, 0), -1)
    cv2.circle(panel, tuple(map(round, coords[weights.argmax()])), 12, (0, 0, 255), -1)
    cv2.putText(panel, level["name"], (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return panel


def render(path, image, memory, attention, gt, origins):
    with rasterio.open(path) as source:
        dop = source.read((1, 2, 3), resampling=Resampling.bilinear)
    scale = np.iinfo(dop.dtype).max if np.issubdtype(dop.dtype, np.integer) else 1.0
    dop = (np.clip(dop.astype(np.float32) / scale, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    dop = cv2.cvtColor(dop, cv2.COLOR_RGB2BGR)
    panels = [cv2.resize(cv2.cvtColor((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR), (dop.shape[1], dop.shape[0]))]
    for name in ("L0", "L1", "L2"):
        level = {**memory[name], "name": name}
        panels.append(attention_panel(dop, level, attention[name], gt, origins))
    combined = cv2.hconcat(panels)
    combined = cv2.resize(combined, (1600, round(combined.shape[0] * 1600 / combined.shape[1])))
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

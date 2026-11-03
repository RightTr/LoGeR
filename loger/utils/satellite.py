"""Windowed access to large RGB GeoTIFF satellite imagery."""

import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import torch.nn.functional as F


class SatelliteTileManager:
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]

    def __init__(self, satellite_path, tile_size=518, tile_stride=448, window_size=3):
        self.satellite_path = satellite_path
        self.tile_size = int(tile_size)
        self.tile_stride = int(tile_stride)
        self.window_size = int(window_size)
        self._dataset = None

    def _open(self):
        if self._dataset is None or self._dataset.closed:
            self._dataset = rasterio.open(self.satellite_path)
        return self._dataset

    def get_tiles(self, center_x, center_y):
        """Read a centered tile grid; center coordinates are raster column and row."""
        center_col = int(round(float(center_x)))
        center_row = int(round(float(center_y)))
        center_left = center_col - self.tile_size // 2
        center_top = center_row - self.tile_size // 2
        half = self.window_size // 2

        origins = []
        for grid_row in range(self.window_size):
            for grid_col in range(self.window_size):
                left = center_left + (grid_col - half) * self.tile_stride
                top = center_top + (grid_row - half) * self.tile_stride
                origins.append((left, top))
        return self._read_tiles(origins)

    def get_all_tiles(self):
        """Read a fixed grid covering the entire DOP."""
        dataset = self._open()
        origins = [(x, y) for y in range(0, dataset.height, self.tile_stride)
                   for x in range(0, dataset.width, self.tile_stride)]
        return self._read_tiles(origins)

    def _read_tiles(self, origins):
        dataset = self._open()
        tile_arrays, pixel_masks = [], []
        for left, top in origins:
            window = Window(left, top, self.tile_size, self.tile_size)
            source = dataset.read(
                indexes=(1, 2, 3), window=window, boundless=True,
                masked=True, fill_value=0,
            )
            raw = np.asarray(source.filled(0))
            masked = np.ma.getmaskarray(source)
            valid = (~masked).all(axis=0) & np.isfinite(raw).all(axis=0)
            scale = np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype, np.integer) else 1.0
            rgb = np.clip(np.nan_to_num(raw.astype(np.float32)) / scale, 0.0, 1.0)
            tile_arrays.append((rgb - self.mean) / self.std)
            pixel_masks.append(valid)

        tiles = torch.from_numpy(np.stack(tile_arrays)).float()
        pixel_valid = torch.from_numpy(np.stack(pixel_masks)).bool()
        origins = torch.tensor(origins, dtype=torch.int64)

        patch_size = 14
        grid_size = self.tile_size // patch_size
        axis = (torch.arange(grid_size, dtype=torch.float32) + 0.5) * patch_size
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        local_coords = torch.stack((xx, yy), dim=-1)
        patch_map_coords = origins[:, None, None, :].float() + local_coords[None, :, :, :]
        patch_map_coords = patch_map_coords.reshape(len(origins), grid_size**2, 2)

        patch_valid = F.avg_pool2d(pixel_valid.float().unsqueeze(1), patch_size, patch_size).squeeze(1) > 0
        patch_valid = patch_valid.flatten(1)
        pooled_valid = F.adaptive_max_pool2d(
            patch_valid.reshape(-1, 1, grid_size, grid_size).float(), (16, 16)
        ).squeeze(1).bool()
        pooled_valid = pooled_valid.flatten(1)
        pooled_coords = F.adaptive_avg_pool2d(
            patch_map_coords.reshape(-1, grid_size, grid_size, 2).permute(0, 3, 1, 2), (16, 16)
        ).permute(0, 2, 3, 1).reshape(len(origins), 16 * 16, 2)

        return {
            "tiles": tiles,
            "tile_origins": origins,
            "pixel_valid_mask": pixel_valid,
            "patch_map_coords": patch_map_coords,
            "pool_map_coords": pooled_coords,
            "patch_valid_mask": patch_valid,
            "pool_valid_mask": pooled_valid,
            "dop_size": torch.tensor((dataset.width, dataset.height), dtype=torch.float32),
        }

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

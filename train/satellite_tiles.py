"""MovingDrone_Tiles data and model helpers for RoMa Hidden Adapter."""

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loger.models.pi3 import Pi3


class TileWindows(Dataset):
    def __init__(self, paths, frames, stride, image_size, satellite_image):
        self.paths = [Path(path) for path in paths]
        self.frames = int(frames)
        self.stride = int(stride)
        self.image_size = tuple(image_size)
        self.satellite_image = satellite_image
        self.rows, self.captures, self.tiles = {}, {}, {}
        self.entries = []
        for sequence in self.paths:
            with (sequence / "poses.csv").open(newline="") as file:
                rows = {int(row["frame_id"]): row for row in csv.DictReader(file)}
            self.rows[sequence] = rows
            frame_ids = sorted(rows)
            valid = set(frame_ids)
            for start in frame_ids[::self.stride]:
                ids = tuple(range(start, start + self.frames))
                if all(frame_id in valid for frame_id in ids):
                    self.entries.append((sequence, ids))

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["captures"] = {}
        return state

    def close(self):
        for capture, _ in self.captures.values():
            capture.release()
        self.captures.clear()

    def _read_rgb(self, sequence, frame_ids):
        sequence_key = str(sequence)
        if sequence_key not in self.captures:
            self.captures[sequence_key] = [cv2.VideoCapture(str(sequence / "video.mp4")), None]
        capture, next_id = self.captures[sequence_key]
        if next_id != frame_ids[0]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ids[0])
        out_h, out_w = self.image_size
        images = []
        for frame_id in frame_ids:
            if self.captures[sequence_key][1] != frame_id:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video decode failed: {sequence} frame={frame_id}")
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div_(255))
        self.captures[sequence_key][1] = frame_ids[-1] + 1
        return torch.stack(images)

    def _read_tile(self, sequence):
        key = str(sequence)
        if key not in self.tiles:
            path = sequence / self.satellite_image
            with Image.open(path) as src:
                image = src.convert("RGB")
                width, height = image.size
                scale = 518 / max(width, height)
                out_h = max(14, round(height * scale / 14) * 14)
                out_w = max(14, round(width * scale / 14) * 14)
                image = image.resize((out_w, out_h), Image.Resampling.BICUBIC)
                self.tiles[key] = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)
        return self.tiles[key]

    def _pose(self, row):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = Rotation.from_quat(
            [float(row[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix().astype(np.float32)
        pose[:3, 3] = [float(row[k]) for k in ("x", "y", "z")]
        return pose

    def load_ids(self, sequence, frame_ids, geometry=True):
        rows = self.rows[sequence]
        images = self._read_rgb(sequence, frame_ids)
        pose_c2w = torch.from_numpy(np.stack([self._pose(rows[i]) for i in frame_ids]))
        poses = torch.linalg.inv(pose_c2w[:1]) @ pose_c2w
        sample = {"images": images, "poses": poses,
                  "frame_ids": torch.tensor(frame_ids, dtype=torch.long),
                  "gt_centers": pose_c2w[:, :3, 3].clone(),
                  "tile": self._read_tile(sequence),
                  "sequence": str(sequence)}
        if not geometry:
            return sample

        raw_K = json.loads((sequence / "intrinsics.json").read_text())
        native_w, native_h = int(raw_K["width"]), int(raw_K["height"])
        out_h, out_w = self.image_size
        sx, sy = out_w / native_w, out_h / native_h
        K = torch.tensor([[float(raw_K["fx"]) * sx, 0., (float(raw_K["cx"]) + .5) * sx - .5],
                          [0., float(raw_K["fy"]) * sy, (float(raw_K["cy"]) + .5) * sy - .5],
                          [0., 0., 1.]], dtype=torch.float32)
        depths = []
        depth_dir = sequence / "depth"
        scale = float(json.loads((depth_dir / "metadata.json").read_text())["scale_m_per_unit"])
        for frame_id in frame_ids:
            raw = cv2.imread(str(depth_dir / f"depth_{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(depth_dir / f"depth_{frame_id:06d}.png")
            depth = raw.astype(np.float32) * scale
            depth[raw == 0] = 0
            depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            depths.append(torch.from_numpy(depth.copy()))
        sample.update(depth=torch.stack(depths), intrinsics=K.expand(len(frame_ids), -1, -1).clone())
        return sample

    def __getitem__(self, index):
        sequence, frame_ids = self.entries[index]
        return self.load_ids(sequence, frame_ids)


def make_loader(paths, frames, stride, size, shuffle, satellite_image, seed=42):
    dataset = TileWindows(paths, frames, stride, size, satellite_image)
    generator = torch.Generator().manual_seed(int(seed))
    return dataset, DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0,
                               pin_memory=torch.cuda.is_available(), generator=generator)


def build_model(config, device):
    options = yaml.safe_load(Path(config["model_config"]).read_text())["model"]
    options.update(roma_enabled=True,
                   roma_checkpoint=config["matchanything_checkpoint"],
                   roma_dino_checkpoint=config["roma_dino_checkpoint"],
                   roma_gradient_checkpointing=True)
    model = Pi3(**options)
    checkpoint = torch.load(config["loger_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.requires_grad_(False)
    model.roma_loger_fusion.requires_grad_(True)
    model = model.to(device)
    model.train()
    return model


def cached_tile_inputs(model, batch, device, cache):
    tile_key = batch["sequence"][0]
    if tile_key not in cache:
        tile = batch["tile"].to(device)
        with torch.no_grad():
            normalized = (tile - model.image_mean) / model.image_std
            roma_target = model.roma_matcher.encode_target(normalized)
        cache[tile_key] = _map_tensors(roma_target, "cpu")
    return _map_tensors(cache[tile_key], device)


def _map_tensors(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _map_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, device) for item in value]
    return value


def forward_batch(model, batch, device, cache, config, fusion_mode="hidden"):
    images = batch["images"].to(device, non_blocking=True)
    tile = batch["tile"].to(device, non_blocking=True)
    roma_target = (cached_tile_inputs(model, batch, device, cache)
                   if fusion_mode == "hidden" else None)
    return model(images, roma_tile_image=tile,
                 roma_match_target_features=roma_target,
                 roma_fusion_mode=fusion_mode,
                 window_size=config["window_size"], overlap_size=config["overlap_size"],
                 num_iterations=1, no_detach=True)


def move_targets(batch, device):
    return {key: batch[key].to(device, non_blocking=True)
            for key in ("depth", "intrinsics", "poses")}


def evaluate_loss(model, loader, device, cache, criterion, config, fusion_mode, max_steps):
    dataset = loader.dataset
    if isinstance(dataset, TileWindows) and max_steps > 0 and max_steps < len(dataset):
        by_sequence = {}
        for index, (sequence, _) in enumerate(dataset.entries):
            by_sequence.setdefault(sequence, []).append(index)
        chosen = []
        sequences = list(by_sequence)
        base, extra = divmod(max_steps, len(sequences))
        for sequence_index, sequence in enumerate(sequences):
            indices = by_sequence[sequence]
            count = min(len(indices), base + (sequence_index < extra))
            if count:
                positions = np.linspace(0, len(indices) - 1, count, dtype=int)
                chosen.extend(indices[position] for position in positions)
        subset = Subset(dataset, chosen)
        loader = DataLoader(subset, batch_size=loader.batch_size or 1,
                            shuffle=False, num_workers=loader.num_workers,
                            pin_memory=loader.pin_memory)
    model.eval()
    totals = {"total": 0., "local": 0., "pose": 0., "global": 0.}
    count = 0
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                pred = forward_batch(model, batch, device, cache, config, fusion_mode)
                losses = criterion(pred, move_targets(batch, device))
            for key in totals:
                totals[key] += float(losses[key])
            count += 1
            if max_steps > 0 and count >= max_steps:
                break
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}



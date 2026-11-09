"""Small two-stage MovingDrone_Tiles experiment for recurrent Satellite Memory."""

import argparse
import csv
import json
import random
from pathlib import Path
import sys
import time
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.uav_satelitte.run_movingdrone import sim3
from loger.models.pi3 import Pi3
from train.loss import LoGeRLoss


class TileWindows(Dataset):
    def __init__(self, paths, frames, stride, image_size, satellite_image):
        self.paths = [Path(path) for path in paths]
        self.frames = int(frames)
        self.stride = int(stride)
        self.image_size = tuple(image_size)
        self.satellite_image = satellite_image
        self.rows, self.captures, self.tiles = {}, {}, {}
        self.entries = []
        self.ids = {str(path): i for i, path in enumerate(sorted(self.paths))}
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
                  "tile_id": torch.tensor(self.ids[str(sequence)], dtype=torch.long),
                  "tile_bbox": torch.tensor([0., 0., 1., 1.]),
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


def make_splits(root, seed, train_fraction, output):
    paths = sorted(path for scene in Path(root).iterdir() if scene.is_dir()
                   for path in scene.iterdir()
                   if path.is_dir() and (path / "video.mp4").is_file()
                   and (path / "poses.csv").is_file() and (path / "depth").is_dir())
    random.Random(seed).shuffle(paths)
    count = max(1, min(len(paths) - 1, round(len(paths) * train_fraction)))
    split = {"seed": seed, "train": [str(x) for x in paths[:count]],
             "val": [str(x) for x in paths[count:]]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "split.json").write_text(json.dumps(split, indent=2))
    return split


def make_loader(paths, frames, stride, size, shuffle, satellite_image):
    dataset = TileWindows(paths, frames, stride, size, satellite_image)
    return dataset, DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0,
                               pin_memory=torch.cuda.is_available())


def build_model(config, device):
    options = yaml.safe_load(Path(config["model_config"]).read_text())["model"]
    options.update(roma_enabled=True,
                   roma_checkpoint=config["matchanything_checkpoint"],
                   roma_dino_checkpoint=config["roma_dino_checkpoint"],
                   roma_gradient_checkpointing=True,
                   roma_probabilistic_topk=config.get("probabilistic_topk", 8))
    model = Pi3(**options)
    checkpoint = torch.load(config["loger_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.requires_grad_(False)
    model.satellite_probabilistic.requires_grad_(True)
    model = model.to(device)
    model.train()
    return model


def cached_tile_inputs(model, batch, device, cache):
    tile_key = batch["sequence"][0]
    if tile_key not in cache:
        tile = batch["tile"].to(device)
        with torch.no_grad():
            sat_features = model.encode_roma_satellite(tile)
            normalized = (tile - model.image_mean) / model.image_std
            roma_target = model.roma_matcher.encode_target(normalized)
        cache[tile_key] = (_map_tensors(sat_features, "cpu"),
                           _map_tensors(roma_target, "cpu"))
    return tuple(_map_tensors(value, device) for value in cache[tile_key])


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


def forward_batch(model, batch, device, cache, config, memory_mode="recurrent",
                  memory=None, new_frame_mask=None):
    images = batch["images"].to(device, non_blocking=True)
    tile = batch["tile"].to(device, non_blocking=True)
    sat_features, roma_target = cached_tile_inputs(model, batch, device, cache)
    return model(images, roma_tile_image=tile,
                 roma_tile_id=batch["tile_id"].to(device),
                 roma_tile_bbox=batch["tile_bbox"].to(device),
                 roma_satellite_features=sat_features,
                 roma_match_target_features=roma_target,
                 roma_memory=memory,
                 roma_new_frame_mask=new_frame_mask,
                 roma_memory_mode=memory_mode,
                 window_size=config["window_size"], overlap_size=config["overlap_size"],
                 num_iterations=1, no_detach=True)


def move_targets(batch, device):
    return {key: batch[key].to(device, non_blocking=True)
            for key in ("depth", "intrinsics", "poses")}


def evaluate_loss(model, loader, device, cache, criterion, config, mode, max_steps):
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
                pred = forward_batch(model, batch, device, cache, config, mode)
                losses = criterion(pred, move_targets(batch, device))
            for key in totals:
                totals[key] += float(losses[key])
            count += 1
            if max_steps > 0 and count >= max_steps:
                break
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def grad_norm(module):
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.


def save_checkpoint(path, model, optimizer, step, phase, val_loss, config):
    torch.save({"satellite_probabilistic": model.satellite_probabilistic.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "phase": phase,
                "val_loss": val_loss, "config": config}, path)


def train_full_trajectories(model, dataset, val_loader, device, config, writer,
                            optimizer, cache, criterion, out, epochs=1):
    """Train in chronological windows, carrying each tile's satellite memory."""
    window = int(config["window_size"])
    overlap = int(config["overlap_size"])
    stride = window - overlap
    step = 0
    best = float("inf")
    model.train()
    for epoch in range(epochs):
        for sequence in dataset.paths:
            frame_ids = sorted(dataset.rows[sequence])
            starts = list(range(0, len(frame_ids) - window + 1, stride))
            last = len(frame_ids) - window
            if starts[-1] != last:
                starts.append(last)
            memory = None
            previous_end = -1
            for start in starts:
                ids = frame_ids[start:start + window]
                sample = dataset.load_ids(sequence, ids, geometry=True)
                batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
                         for key, value in sample.items()}
                fresh = torch.tensor([[frame_id > previous_end for frame_id in ids]],
                                     dtype=torch.bool, device=device)
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    pred = forward_batch(model, batch, device, cache, config,
                                         "recurrent", memory, fresh)
                    losses = criterion(pred, move_targets(batch, device))
                loss = losses["total"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite full-trajectory loss at {sequence} frame {ids[0]}")
                loss.backward()
                module_grad = torch.nn.utils.clip_grad_norm_(
                    model.satellite_probabilistic.parameters(), 1.0)
                optimizer.step()
                memory = {key: value.detach() if torch.is_tensor(value) else value
                          for key, value in pred["satellite_memory"].items()}
                previous_end = ids[-1]
                step += 1
                for key in ("total", "local", "pose", "global"):
                    writer.add_scalar(f"full_trajectory/train_{key}", losses[key].detach(), step)
                writer.add_scalar("full_trajectory/fusion_grad_norm", float(module_grad), step)
                writer.add_scalar("full_trajectory/memory_update_norm",
                                  pred["satellite_memory_update_norm"], step)
                writer.add_scalar("full_trajectory/memory_coverage",
                                  pred["satellite_memory_coverage"], step)
                writer.add_scalar("full_trajectory/roma_match_probability",
                                  pred["roma_match_probability"], step)
                if step % 25 == 0:
                    print(f"epoch={epoch + 1}/{epochs} step={step} "
                          f"sequence={sequence.parent.name}/{sequence.name} "
                          f"frames={ids[0]}-{ids[-1]} loss={float(loss):.5f} "
                          f"memory_update={float(pred['satellite_memory_update_norm']):.4g} "
                          f"grad={float(module_grad):.4g}", flush=True)
                if step % 100 == 0:
                    save_checkpoint(out / "latest.pt", model, optimizer, step,
                                    "full_trajectory", float(loss), config)
                del pred, losses, batch, sample
            print(f"Finished full track {sequence.parent.name}/{sequence.name}: "
                  f"{len(frame_ids)} frames, {len(starts)} ordered windows", flush=True)
        val = evaluate_full_trajectories(model, val_loader.dataset, device, cache,
                                         config, criterion)
        writer.add_scalar("full_trajectory/val_total", val["total"], epoch + 1)
        writer.add_scalar("full_trajectory/val_pointmap", val["local"], epoch + 1)
        writer.add_scalar("full_trajectory/val_pose", val["pose"], epoch + 1)
        print(f"Full-trajectory epoch {epoch + 1}: validation {val}", flush=True)
        save_checkpoint(out / "latest.pt", model, optimizer, step,
                        "full_trajectory", val["total"], config)
        if val["total"] < best:
            best = val["total"]
            save_checkpoint(out / "best.pt", model, optimizer, step,
                            "full_trajectory", best, config)
    return step, best


def evaluate_full_trajectories(model, dataset, device, cache, config, criterion,
                               mode="recurrent"):
    """Evaluate each held-out tile in order with memory carried to its end."""
    was_training = model.training
    model.eval()
    totals = {key: 0. for key in ("total", "local", "pose", "global")}
    windows = 0
    window = int(config["window_size"])
    stride = window - int(config["overlap_size"])
    with torch.no_grad():
        for sequence in dataset.paths:
            frame_ids = sorted(dataset.rows[sequence])
            starts = list(range(0, len(frame_ids) - window + 1, stride))
            last = len(frame_ids) - window
            if starts[-1] != last:
                starts.append(last)
            memory = None
            previous_end = -1
            for start in starts:
                ids = frame_ids[start:start + window]
                sample = dataset.load_ids(sequence, ids, geometry=True)
                batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
                         for key, value in sample.items()}
                fresh = torch.tensor([[frame_id > previous_end for frame_id in ids]],
                                     dtype=torch.bool, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    pred = forward_batch(model, batch, device, cache, config,
                                         mode, memory, fresh)
                    losses = criterion(pred, move_targets(batch, device))
                for key in totals:
                    totals[key] += float(losses[key])
                if mode != "off":
                    memory = {key: value.detach() if torch.is_tensor(value) else value
                              for key, value in pred["satellite_memory"].items()}
                previous_end = ids[-1]
                windows += 1
                del pred, losses, batch, sample
            print(f"Validated full track {sequence.parent.name}/{sequence.name}: "
                  f"{len(frame_ids)} frames, {len(starts)} ordered windows", flush=True)
    if was_training:
        model.train()
    return {key: value / max(windows, 1) for key, value in totals.items()}


def train_phase(model, train_loader, val_loader, device, config, writer, optimizer,
                phase, steps, start_step, cache, criterion, best, out, deadline=None):
    data = iter(train_loader)
    interval = int(config["validate_every"])
    val_steps = int(config["val_steps"])
    memory_mode = "reset" if phase == "single_chunk" else "recurrent"
    local_step = 0
    while local_step < steps and (deadline is None or time.monotonic() < deadline):
        local_step += 1
        try:
            batch = next(data)
        except StopIteration:
            data = iter(train_loader)
            batch = next(data)
        step = start_step + local_step
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            pred = forward_batch(model, batch, device, cache, config, memory_mode)
            losses = criterion(pred, move_targets(batch, device))
        loss = losses["total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {phase} loss at step {step}: {batch['sequence'][0]}")
        loss.backward()
        module_grad = torch.nn.utils.clip_grad_norm_(
            model.satellite_probabilistic.parameters(), 1.0)
        memory_grad = grad_norm(model.satellite_probabilistic.write_delta)
        optimizer.step()
        for key in ("total", "local", "pose", "global"):
            writer.add_scalar(f"{phase}/train_{key}", losses[key].detach(), step)
        writer.add_scalar(f"{phase}/fusion_grad_norm", float(module_grad), step)
        writer.add_scalar(f"{phase}/memory_write_grad_norm", memory_grad, step)
        output_layer = model.satellite_probabilistic.read_out
        writer.add_scalar(f"{phase}/output_weight_norm",
                          output_layer.weight.detach().float().norm(), step)
        writer.add_scalar(f"{phase}/roma_certainty", pred["roma_certainty"], step)
        writer.add_scalar(f"{phase}/roma_match_probability", pred["roma_match_probability"], step)
        if "satellite_probabilistic_coverage" in pred:
            writer.add_scalar(f"{phase}/probabilistic_coverage",
                              pred["satellite_probabilistic_coverage"], step)
            writer.add_scalar(f"{phase}/probabilistic_write_mass",
                              pred["satellite_probabilistic_write_mass"], step)
            writer.add_scalar(f"{phase}/probabilistic_delta_ratio",
                              pred["satellite_probabilistic_delta_ratio"], step)
        writer.add_scalar(f"{phase}/roma_reliable_patch_fraction",
                          pred["roma_reliable_patch_fraction"], step)
        writer.add_scalar(f"{phase}/memory_update_norm",
                          pred["satellite_memory_update_norm"], step)
        writer.add_scalar(f"{phase}/memory_coverage", pred["satellite_memory_coverage"], step)
        writer.add_scalar(f"{phase}/gpu_peak_gb",
                          torch.cuda.max_memory_allocated(device) / 1024**3
                          if device.type == "cuda" else 0., step)

        time_up = deadline is not None and time.monotonic() >= deadline
        if local_step % interval == 0 or local_step == steps or time_up:
            val = evaluate_loss(model, val_loader, device, cache, criterion,
                                config, memory_mode, val_steps)
            writer.add_scalar(f"{phase}/val_total", val["total"], step)
            writer.add_scalar(f"{phase}/val_pointmap", val["local"], step)
            writer.add_scalar(f"{phase}/val_pose", val["pose"], step)
            print(f"{phase} step={local_step}/{steps} train={float(loss):.5f} "
                  f"val={val['total']:.5f} local={val['local']:.5f} pose={val['pose']:.5f} "
                  f"grad_fusion={float(module_grad):.4g} grad_mem={memory_grad:.4g} "
                  f"coverage={float(pred['satellite_memory_coverage']):.3f}", flush=True)
            save_checkpoint(out / "latest.pt", model, optimizer, step, phase, val["total"], config)
            if val["total"] < best:
                best = val["total"]
                save_checkpoint(out / "best.pt", model, optimizer, step, phase, best, config)
            model.train()
    return start_step + local_step, best


def evaluate_trajectory(model, dataset, sequence, device, cache, config, mode, count=64):
    rows = dataset.rows[sequence]
    frame_ids = np.linspace(min(rows), max(rows), count, dtype=int).tolist()
    sample = dataset.load_ids(sequence, frame_ids, geometry=False)
    batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
             for key, value in sample.items()}
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        pred = forward_batch(model, batch, device, cache, config, mode)
    centers = pred["camera_poses"][0, :, :3, 3].float().cpu().numpy().astype(np.float64)
    gt = sample["gt_centers"].numpy().astype(np.float64)
    scale, rotation, translation = sim3(centers, gt)
    aligned = scale * (centers @ rotation.T) + translation
    errors = np.linalg.norm(aligned - gt, axis=1)
    rpe = np.linalg.norm(np.diff(aligned, axis=0) - np.diff(gt, axis=0), axis=1)
    return aligned, gt, {"ate_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
                         "ate_mean_m": float(errors.mean()),
                         "relative_translation_rmse_m": float(np.sqrt(np.mean(rpe ** 2)))}


def final_evaluation(model, val_dataset, device, cache, criterion, config, out, writer):
    modes = ("off", "reset", "recurrent")
    names = {"off": "baseline", "reset": "no_memory",
             "recurrent": "recurrent_memory"}
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics = {"geometry_loss": {}, "trajectories": {}}
    for mode in modes:
        result = evaluate_loss(model, loader, device, cache, criterion, config,
                              mode, min(len(loader), config["final_loss_windows"]))
        metrics["geometry_loss"][names[mode]] = result
        print(f"validation {names[mode]}: {result}", flush=True)
        for key, value in result.items():
            writer.add_scalar(f"final/{names[mode]}_{key}", value, 0)

    fig, axes = plt.subplots(len(val_dataset.paths), 1, figsize=(8, 5 * len(val_dataset.paths)),
                             squeeze=False, constrained_layout=True)
    colors = {"baseline": "#4c78a8", "probabilistic": "#e45756", "no_memory": "#f58518",
              "recurrent_memory": "#54a24b", "shuffled_memory": "#e45756"}
    for row, sequence in enumerate(val_dataset.paths):
        ax = axes[row, 0]
        scene_name = sequence.parent.name + "/" + sequence.name
        metrics["trajectories"][scene_name] = {}
        for mode in modes:
            aligned, gt, values = evaluate_trajectory(
                model, val_dataset, sequence, device, cache, config, mode)
            key = names[mode]
            metrics["trajectories"][scene_name][key] = values
            ax.plot(aligned[:, 0], aligned[:, 1], color=colors[key], linewidth=1.8,
                    label=f"{key} ATE {values['ate_rmse_m']:.1f}m")
            if mode == "off":
                ax.plot(gt[:, 0], gt[:, 1], color="#111111", linewidth=1.4,
                        linestyle="--", label="GT")
        ax.set_title(scene_name + " | 64 sampled frames")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.axis("equal"); ax.grid(alpha=.25); ax.legend()
    fig.savefig(out / "validation_trajectories.png", dpi=160)
    writer.add_figure("final/validation_trajectories", fig, 0)
    plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train(config, resume=False, resume_from=None, split_file=None,
          duration_minutes=None, output=None, full_trajectory_epochs=0):
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    root = Path(config["dataset_root"])
    out = Path(output or config["output"])
    config["output"] = str(out)
    out.mkdir(parents=True, exist_ok=True)
    if resume_from:
        resume_from = Path(resume_from)
        if not (out / "latest.pt").exists():
            shutil.copy2(resume_from, out / "latest.pt")
        previous_best = resume_from.with_name("best.pt")
        if previous_best.is_file() and not (out / "best.pt").exists():
            shutil.copy2(previous_best, out / "best.pt")
        previous_split = resume_from.with_name("split.json")
        if previous_split.is_file():
            shutil.copy2(previous_split, out / "split.json")
        resume = True
    output_split = out / "split.json"
    if split_file:
        source_split = Path(split_file)
        shutil.copy2(source_split, output_split)
        split = json.loads(source_split.read_text())
    elif config.get("split_file") and Path(config["split_file"]).is_file():
        shutil.copy2(config["split_file"], output_split)
        split = json.loads(output_split.read_text())
    elif resume_from and output_split.is_file():
        split = json.loads(output_split.read_text())
    else:
        split = make_splits(root, seed, float(config["train_fraction"]), out)
    size = tuple(config["image_size"])
    satellite_image = config["satellite_image"]
    train6, train6_loader = make_loader(split["train"], 6, config["sample_stride"], size, True, satellite_image)
    val6, val6_loader = make_loader(split["val"], 6, config["sample_stride"], size, False, satellite_image)
    train10, train10_loader = make_loader(split["train"], 10, config["sample_stride"], size, True, satellite_image)
    val10, val10_loader = make_loader(split["val"], 10, config["sample_stride"], size, False, satellite_image)
    print(f"tile sequences: {len(split['train'])} train / {len(split['val'])} val; "
          f"windows: {len(train6)} single, {len(train10)} paired", flush=True)

    device = torch.device(config["device"])
    model = build_model(config, device)
    if config.get("init_checkpoint"):
        initial = torch.load(config["init_checkpoint"], map_location="cpu", weights_only=True)
        model.satellite_probabilistic.load_state_dict(initial["satellite_probabilistic"])
        print(f"Initialized probabilistic fusion from {config['init_checkpoint']} "
              f"(step {initial.get('step', '?')})", flush=True)
    criterion = LoGeRLoss(config["lambda_global"], config["lambda_rotation"],
                          config["lambda_translation"])
    cache = {}
    writer = SummaryWriter(out / "tensorboard")
    baseline = evaluate_loss(model, val6_loader, device, cache, criterion, config,
                             "off", config["val_steps"])
    print(f"Pure LoGeR validation baseline: {baseline}", flush=True)
    writer.add_scalar("baseline/val_total", baseline["total"], 0)

    optimizer = torch.optim.AdamW(
        model.satellite_probabilistic.parameters(),
        lr=config.get("probabilistic_lr", config.get("bidirectional_lr", 1e-4)),
        weight_decay=config["weight_decay"])
    best = float("inf")
    start_phase = "single_chunk"
    step = 0
    resume_path = out / "latest.pt"
    if resume and resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.satellite_probabilistic.load_state_dict(saved["satellite_probabilistic"])
        optimizer.load_state_dict(saved["optimizer"])
        step = int(saved["step"])
        start_phase = saved["phase"]
        print(f"Resuming {start_phase} from step {step}", flush=True)
    deadline = (time.monotonic() + duration_minutes * 60
                if duration_minutes else None)
    if full_trajectory_epochs:
        step, best = train_full_trajectories(
            model, train6, val6_loader, device, config, writer, optimizer,
            cache, criterion, out, epochs=full_trajectory_epochs)
        best_state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        model.satellite_probabilistic.load_state_dict(best_state["satellite_probabilistic"])
        baseline_val = evaluate_full_trajectories(model, val6, device, cache,
                                                  config, criterion, mode="off")
        recurrent_val = evaluate_full_trajectories(model, val6, device, cache,
                                                   config, criterion, mode="recurrent")
        (out / "full_trajectory_validation.json").write_text(json.dumps(
            {"baseline": baseline_val, "probabilistic_recurrent": recurrent_val,
             "best_step": best_state["step"], "best_val_loss": best_state["val_loss"]},
            indent=2))
        print(f"Full-track validation: baseline={baseline_val}; "
              f"probabilistic recurrent={recurrent_val}", flush=True)
        writer.close()
        train6.close(); val6.close(); train10.close(); val10.close()
        print(f"Finished full-trajectory training. Best step={best_state['step']} "
              f"val={best_state['val_loss']:.5f}. Results: {out}", flush=True)
        return
    if start_phase == "single_chunk" and step < config["single_steps"]:
        step, best = train_phase(model, train6_loader, val6_loader, device, config,
                                 writer, optimizer, "single_chunk", config["single_steps"] - step,
                                 step, cache, criterion, best, out, deadline)
    completed_two = max(0, step - config["single_steps"])
    if duration_minutes:
        if start_phase == "two_chunks" and (out / "best.pt").is_file():
            best_state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
            best = float(best_state.get("val_loss", float("inf")))
        else:
            best = float("inf")
        if deadline is None or time.monotonic() < deadline:
            step, best = train_phase(model, train10_loader, val10_loader, device, config,
                                     writer, optimizer, "two_chunks", 10**9,
                                     step, cache, criterion, best, out, deadline)
    elif start_phase != "two_chunks" or completed_two < config["two_chunk_steps"]:
        best = float("inf")
        step, best = train_phase(model, train10_loader, val10_loader, device, config,
                                 writer, optimizer, "two_chunks",
                                 config["two_chunk_steps"] - completed_two,
                                 step, cache, criterion, best, out)

    best_state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.satellite_probabilistic.load_state_dict(best_state["satellite_probabilistic"])
    final_evaluation(model, val6, device, cache, criterion, config, out, writer)
    writer.close()
    train6.close(); val6.close(); train10.close(); val10.close()
    print(f"Finished. Results: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/satellite_memory_tiles.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--split-file")
    parser.add_argument("--duration-minutes", type=float)
    parser.add_argument("--output")
    parser.add_argument("--full-trajectory-epochs", type=int, default=0)
    args = parser.parse_args()
    train(yaml.safe_load(Path(args.config).read_text()), resume=args.resume,
          resume_from=args.resume_from, duration_minutes=args.duration_minutes,
          split_file=args.split_file, output=args.output,
          full_trajectory_epochs=args.full_trajectory_epochs)


if __name__ == "__main__":
    main()

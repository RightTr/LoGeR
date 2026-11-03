"""MovingDrone dataset with lazy RGB, pose, and geometry outputs."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from scipy.spatial.transform import Rotation

from .base import BaseSequenceDataset


class MovingDroneDataset(BaseSequenceDataset):
    def __init__(self, root, variant="baseline", output="rgb", dop_year="2023", **kwargs):
        if output not in ("rgb", "pose", "geometry"):
            raise ValueError("output must be 'rgb', 'pose', or 'geometry'")
        self.output = output
        self.dop_year = str(dop_year)
        self.scene_cache = {}
        scenes = kwargs.pop("scenes", None)
        if not scenes:
            raise ValueError("Pass configured scene names with scenes=(...)")
        sequence_paths = [Path(root) / scene / str(variant) for scene in scenes]
        super().__init__(sequence_paths, **kwargs)

    def _index_frame_ids(self, sequence, video_frame_count):
        if self.output == "rgb":
            return list(range(video_frame_count))
        return [frame_id for frame_id in sorted(self.load_poses(sequence))
                if 0 <= frame_id < video_frame_count]

    def load_poses(self, sequence):
        cache = self.scene_cache.setdefault(sequence, {})
        if "poses" not in cache:
            with (sequence / "poses.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            poses = {}
            for row in rows:
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = Rotation.from_quat(
                    [float(row[k]) for k in ("qx", "qy", "qz", "qw")]
                ).as_matrix().astype(np.float32)
                pose[:3, 3] = [float(row[k]) for k in ("x", "y", "z")]
                quaternion = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")], np.float32)
                poses[int(row["frame_id"])] = (pose, quaternion)
            cache["poses"] = poses
        return cache["poses"]

    def load_depths(self, sequence, frame_id):
        path = sequence / "depth" / f"depth_{frame_id:04d}.png"
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Depth requested but missing directory: {path.parent}")
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Depth requested but frame is missing: {path}")
        raw = raw.astype(np.float32)
        sidecar = json.loads(path.with_suffix(".json").read_text())
        d_min = float(sidecar.get("depth_min_m", 0.0))
        ray = d_min + raw * ((float(sidecar["depth_max_m"]) - d_min) / 65535.0)
        mask = (raw > 0) & np.isfinite(ray)
        ray[~mask] = 0
        out_h, out_w = self.image_size
        ray = cv2.resize(ray, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), (out_w, out_h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        return torch.from_numpy(ray.copy()), torch.from_numpy(mask)

    def load_images(self, sequence, frame_ids):
        if sequence not in self._captures:
            self._captures[sequence] = [cv2.VideoCapture(str(sequence / "video.mp4")), None]
        height, width = self.image_size
        images = []
        for frame_id in frame_ids:
            capture, last_frame_id = self._captures[sequence]
            if last_frame_id != frame_id:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            self._captures[sequence][1] = frame_id + 1 if ok else None
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            image = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy())
            image = image.permute(2, 0, 1).float() / 255
            images.append(self.transform(image) if self.transform is not None else image)
        return torch.stack(images)

    def load_requested_fields(self, sequence, frame_ids):
        if self.output == "rgb":
            return {}
        poses = self.load_poses(sequence)
        pose_mats, quaternions = [], []
        for frame_id in frame_ids:
            pose, quaternion = poses[frame_id]
            pose_mats.append(torch.from_numpy(pose.copy()))
            quaternions.append(torch.from_numpy(quaternion.copy()))
        pose_mats = torch.stack(pose_mats)
        values = {
            "pose_c2w": pose_mats,
            "position_xyz": pose_mats[:, :3, 3],
            "quaternion_wxyz": torch.stack(quaternions),
        }
        if self.output == "pose":
            if self.mode == "frame":
                values = {key: value[0] for key, value in values.items()}
            return values

        ray_depths, masks = zip(*(self.load_depths(sequence, frame_id) for frame_id in frame_ids))
        ray_depths, masks = torch.stack(ray_depths), torch.stack(masks)
        native_h, native_w, raw_K = self.load_intrinsics(sequence)
        out_h, out_w = self.image_size
        sx, sy = out_w / native_w, out_h / native_h
        K = raw_K.clone()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = (K[0, 2] + .5) * sx - .5
        K[1, 2] = (K[1, 2] + .5) * sy - .5

        h, w = self.image_size
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                                torch.arange(w, dtype=torch.float32), indexing="ij")
        pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=0).reshape(3, -1)
        rays = torch.linalg.inv(K) @ pixels
        ray_norm = torch.linalg.vector_norm(rays, dim=0).reshape(h, w)
        z_depths, local_points = [], []
        for ray, mask in zip(ray_depths, masks):
            z = ray / ray_norm
            z[~mask] = 0
            z_depths.append(z)
            local_points.append((rays * z.reshape(1, -1)).T.reshape(h, w, 3))
        z_depths = torch.stack(z_depths)
        local_points = torch.stack(local_points)

        relative_poses = torch.linalg.inv(pose_mats[0]).unsqueeze(0) @ pose_mats
        global_points = torch.stack([
            points @ pose[:3, :3].T + pose[:3, 3]
            for points, pose in zip(local_points, relative_poses)
        ])
        norm_factor = torch.linalg.vector_norm(global_points[masks], dim=-1).mean()
        if not torch.isfinite(norm_factor) or norm_factor <= 0:
            raise ValueError(f"No valid depth points for geometry sample: {sequence}")
        local_points /= norm_factor
        global_points /= norm_factor
        relative_poses[:, :3, 3] /= norm_factor

        dop = self.load_dop(sequence)
        xmin, ymin, xmax, ymax = dop["dop_bounds"].tolist()
        dop_h, dop_w = dop["dop_size_hw"].tolist()
        sx, sy = (dop_w - 1) / (xmax - xmin), (dop_h - 1) / (ymax - ymin)
        world_to_dop = torch.tensor([[sx, 0, -xmin * sx],
                                     [0, -sy, ymax * sy], [0, 0, 1]], dtype=torch.float32)
        values.update(
            depth_ray_m=ray_depths,
            depth_z_m=z_depths,
            K=K.expand(len(frame_ids), -1, -1).clone(),
            valid_mask=masks,
            local_points=local_points,
            global_points=global_points,
            poses=relative_poses,
            gt_norm_factor=norm_factor,
            dop_size_hw=dop["dop_size_hw"],
            world_to_dop=world_to_dop,
        )
        if self.mode == "frame":
            values = {
                key: value if key in ("gt_norm_factor", "dop_size_hw", "world_to_dop") else value[0]
                for key, value in values.items()
            }
        return values

    def load_dop(self, sequence):
        cache = self.scene_cache.setdefault(sequence, {})
        if "dop" not in cache:
            meta = json.loads((sequence / "meta.json").read_text())
            dop = meta["dops"][f"dop_{self.dop_year}"]
            path = sequence / dop["file"]
            with rasterio.open(path) as image:
                size = torch.tensor([image.height, image.width], dtype=torch.long)
            cache["dop"] = {
                "dop_path": str(path),
                "dop_bounds": torch.tensor(dop["bounds"], dtype=torch.float32),
                "dop_size_hw": size,
            }
        return cache["dop"]

    def load_intrinsics(self, sequence):
        cache = self.scene_cache.setdefault(sequence, {})
        if "intrinsics" not in cache:
            intr = json.loads((sequence / "intrinsics.json").read_text())
            cache["intrinsics"] = (
                int(intr["height"]), int(intr["width"]),
                torch.tensor([[intr["fx"], 0, intr["cx"]],
                              [0, intr["fy"], intr["cy"]], [0, 0, 1]], dtype=torch.float32),
            )
        return cache["intrinsics"]

    def get_scene_metadata(self, scene):
        info = super().get_scene_metadata(scene)
        sequence = Path(info["path"])
        info.update(self.load_dop(sequence))
        return info

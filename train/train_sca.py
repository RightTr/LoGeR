"""Train shared satellite SCA with two overlapping MovingDrone chunks."""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.transform import Rotation

from loger.models.pi3 import Pi3
from loger.utils.satellite import SatelliteTileManager


def load_sequences(scene, frames, step):
    with (scene / "poses.csv").open() as source:
        rows = list(csv.DictReader(source))
    video = cv2.VideoCapture(str(scene / "video.mp4"))
    for start in range(0, len(rows) - frames + 1, step):
        selected = rows[start:start + frames]
        images, poses = [], []
        for row in selected:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_id"]))
            ok, image = video.read()
            if not ok:
                break
            image = cv2.cvtColor(cv2.resize(image, (504, 280)), cv2.COLOR_BGR2RGB)
            images.append(torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = Rotation.from_quat([float(row[key]) for key in ("qx", "qy", "qz", "qw")]).as_matrix()
            pose[:3, 3] = [float(row[key]) for key in ("x", "y", "z")]
            poses.append(torch.from_numpy(pose))
        if len(images) == frames:
            yield torch.stack(images)[None], torch.stack(poses)[None]
    video.release()


def relative_pose(poses):
    return torch.linalg.inv(poses[..., :-1, :, :]) @ poses[..., 1:, :, :]


def pose_error(predicted, target):
    rotation = predicted[..., :3, :3].transpose(-1, -2) @ target[..., :3, :3]
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cosine)
    translation = 1 - F.cosine_similarity(predicted[..., :3, 3], target[..., :3, 3], dim=-1)
    return angle.mean() + translation.mean()


def geometry_loss(output, gt, window, overlap):
    poses = output["camera_poses"]
    boundary = window - overlap
    first = pose_error(relative_pose(poses[:, :boundary]), relative_pose(gt[:, :boundary]))
    second = pose_error(relative_pose(poses[:, boundary:]), relative_pose(gt[:, boundary:]))
    return (first + second) / 2


def chunk_loss(output):
    prev = output["overlap_prev_pcd"]
    next_ = output["overlap_next_pcd"]
    scale = (prev * next_).sum(dim=(-3, -2, -1), keepdim=True) / next_.square().sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
    points = F.smooth_l1_loss(prev, scale.detach() * next_)
    cameras = pose_error(relative_pose(output["overlap_prev_cam"]), relative_pose(output["overlap_next_cam"]))
    return points + cameras


def satellite_features(model, path, center):
    with SatelliteTileManager(path) as manager:
        tiles = manager.get_tiles(*center)
    with torch.no_grad():
        return model.encode_satellite(tiles)


def evaluate(model, scene, memory, args, baseline=False):
    model.sca_enabled = not baseline
    pose_errors, chunk_errors = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for images, gt in load_sequences(scene, 2 * args.window - args.overlap, args.window - args.overlap):
            output = model(images.to(args.device), satellite_features=memory, window_size=args.window,
                           overlap_size=args.overlap, no_detach=True)
            pose_errors.append(geometry_loss(output, gt.to(args.device), args.window, args.overlap).item())
            chunk_errors.append(chunk_loss(output).item())
    model.sca_enabled = True
    return float(np.mean(pose_errors)), float(np.mean(chunk_errors)), time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ckpts/LoGeR/original_config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--train-scene", type=Path, required=True)
    parser.add_argument("--val-scene", type=Path, required=True)
    parser.add_argument("--dop", required=True)
    parser.add_argument("--center", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--wrong-dop", required=True)
    parser.add_argument("--wrong-center", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-weight", type=float, default=0.1)
    parser.add_argument("--save", type=Path, default=Path("sca.pt"))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as source:
        config = yaml.safe_load(source)["model"]
    config["sca_enabled"] = True
    model = Pi3(**{key: value for key, value in config.items() if key in Pi3.__init__.__code__.co_varnames})
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    model.to(args.device).eval()
    optimizer = torch.optim.AdamW(model.sca.parameters(), lr=args.lr, weight_decay=0.01)
    correct = satellite_features(model, args.dop, args.center)
    wrong = satellite_features(model, args.wrong_dop, args.wrong_center or args.center)
    best = float("inf")
    frames = 2 * args.window - args.overlap
    for epoch in range(args.epochs):
        losses = []
        for images, gt in load_sequences(args.train_scene, frames, args.window - args.overlap):
            output = model(images.to(args.device), satellite_features=correct, window_size=args.window,
                           overlap_size=args.overlap, no_detach=True)
            loss = geometry_loss(output, gt.to(args.device), args.window, args.overlap)
            loss = loss + args.chunk_weight * chunk_loss(output)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.sca.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        e0 = evaluate(model, args.val_scene, None, args, baseline=True)
        e1 = evaluate(model, args.val_scene, correct, args)
        e2 = evaluate(model, args.val_scene, wrong, args)
        print(f"epoch={epoch + 1} train={np.mean(losses):.4f} E0={e0} E1={e1} E2={e2}")
        score = e1[0] + args.chunk_weight * e1[1]
        if score < best:
            best = score
            torch.save({"sca": model.sca.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1}, args.save)


if __name__ == "__main__":
    main()

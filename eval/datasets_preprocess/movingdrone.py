"""Read sampled video frames and ground-truth poses for MovingDrone evaluation."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch


def find_scenes(root):
    return sorted(
        video.parent.relative_to(root).as_posix()
        for video in root.rglob('video.mp4')
        if (video.parent / 'poses.csv').is_file() and '_chunks' not in video.parts
    )


def group_scenes(names):
    groups = {}
    for name in names:
        groups.setdefault(Path(name).parts[0], []).append(name)
    return groups


def result_dir(output, name, frames):
    parts = Path(name).parts
    return Path(output) / parts[0] / str(frames) / Path(*parts[1:])


def load_frames(scene, count, size=(504, 280)):
    with (scene / 'poses.csv').open() as f:
        rows = list(csv.DictReader(f))
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    video = cv2.VideoCapture(str(scene / 'video.mp4'))
    frames, poses, frame_ids = [], [], []

    for index in np.unique(indices):
        row = rows[index]
        video.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame_id']))
        ok, frame = video.read()
        if ok:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255)
            poses.append([float(row[key]) for key in ('x', 'y', 'z')])
            frame_ids.append(int(row['frame_id']))

    video.release()
    return torch.stack(frames), np.asarray(poses), np.asarray(frame_ids)

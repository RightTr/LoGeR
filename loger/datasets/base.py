"""Shared video indexing and decoding for sequence datasets."""

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class BaseSequenceDataset(Dataset):
    def __init__(self, sequence_paths, mode="window", window_size=8, stride=4,
                 frame_step=1, image_size=(280, 504), frame_range=None, transform=None):
        self.mode = mode
        self.image_size = tuple(image_size)
        self.transform = transform
        self._captures = {}
        self.sequence_fps = {}
        self.index = []
        self.indices_by_scene = {}

        for sequence in sequence_paths:
            sequence = Path(sequence)
            scene = sequence.parent.name
            video_path = sequence / "video.mp4"
            capture = cv2.VideoCapture(str(video_path))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.sequence_fps[sequence] = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            ids = self._index_frame_ids(sequence, frame_count)
            if frame_range is not None:
                start, stop = frame_range
                ids = [frame_id for frame_id in ids if start <= frame_id < stop]
            if not ids:
                continue
            entries = self._make_entries(sequence, ids, mode, window_size, stride, frame_step)
            self.indices_by_scene.setdefault(scene, []).extend(
                range(len(self.index), len(self.index) + len(entries))
            )
            self.index.extend(entries)

    @staticmethod
    def _make_entries(sequence, frame_ids, mode, window_size, stride, frame_step):
        if mode == "frame":
            return [(sequence, (frame_id,)) for frame_id in frame_ids]
        available = set(frame_ids)
        return [
            (sequence, tuple(frame_id + i * frame_step for i in range(window_size)))
            for frame_id in frame_ids[::stride]
            if all(frame_id + i * frame_step in available for i in range(window_size))
        ]

    def __len__(self):
        return len(self.index)

    def get_scene_metadata(self, scene):
        sequence, _ = self.index[self.indices_by_scene[scene][0]]
        return {"name": scene, "path": str(sequence), "variant": sequence.name}

    def __getitem__(self, index):
        sequence, frame_ids = self.index[index]
        fps = self.sequence_fps[sequence]

        sample = {
            "frame_ids": torch.tensor(frame_ids, dtype=torch.long),
            "timestamps": torch.tensor([frame_id / fps for frame_id in frame_ids], dtype=torch.float64),
            "scene": sequence.parent.name,
            "variant": sequence.name,
        }
        sample["rgb"] = self.load_images(sequence, frame_ids)
        sample.update(self.load_requested_fields(sequence, frame_ids))

        if self.mode == "frame":
            for field in ("rgb", "frame_ids", "timestamps"):
                if field in sample and torch.is_tensor(sample[field]):
                    sample[field] = sample[field][0]
        return sample

    def close(self):
        for capture, _ in self._captures.values():
            capture.release()
        self._captures.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_captures"] = {}
        return state

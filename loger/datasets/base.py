"""Sequence indexing and on-demand video decoding for frame and window datasets."""

import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class BaseSequenceDataset(Dataset):
    def __init__(
        self,
        root,
        mode="window",
        window_size=8,
        stride=4,
        frame_step=1,
        image_size=(280, 504),
        scenes=None,
        frame_range=None,
        transform=None,
    ):
        if mode not in ("frame", "window"):
            raise ValueError("mode must be 'frame' or 'window'")
        if min(window_size, stride, frame_step) < 1:
            raise ValueError("window_size, stride and frame_step must be positive")

        self.root = Path(root)
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.frame_step = frame_step
        self.image_size = image_size
        self.transform = transform
        self._captures = {}
        self._capture_positions = {}
        self._capture_pid = os.getpid()

        selected = set(scenes) if scenes is not None else None
        self.sequences = {}
        self.index = []
        for sequence in self._discover_sequences():
            scene = sequence.parent.name
            if selected is not None and scene not in selected:
                continue
            records = self._load_records(sequence)
            if frame_range is not None:
                start, stop = frame_range
                records = [r for r in records if start <= r["frame_id"] < stop]
            if not records:
                continue
            self.sequences[sequence] = records
            if mode == "frame":
                self.index.extend((sequence, (i,)) for i in range(len(records)))
            else:
                span = (window_size - 1) * frame_step + 1
                self.index.extend(
                    (sequence, tuple(range(start, start + span, frame_step)))
                    for start in range(0, len(records) - span + 1, stride)
                )

    def _discover_sequences(self):
        raise NotImplementedError

    def _load_records(self, sequence):
        raise NotImplementedError

    def _load_frame(self, sequence, record):
        raise NotImplementedError

    def __len__(self):
        return len(self.index)

    def _capture(self, sequence):
        if self._capture_pid != os.getpid():
            self.close()
            self._capture_pid = os.getpid()
        if sequence not in self._captures:
            capture = cv2.VideoCapture(str(sequence / "video.mp4"))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {sequence / 'video.mp4'}")
            self._captures[sequence] = capture
            self._capture_positions[sequence] = None
        return self._captures[sequence]

    def _read_rgb(self, sequence, frame_id):
        capture = self._capture(sequence)
        if self._capture_positions[sequence] != frame_id:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        self._capture_positions[sequence] = frame_id + 1 if ok else None
        if not ok:
            raise RuntimeError(f"Cannot decode {sequence} frame_id={frame_id}")
        height, width = self.image_size
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255
        return self.transform(rgb) if self.transform is not None else rgb

    def __getitem__(self, index):
        sequence, positions = self.index[index]
        records = self.sequences[sequence]
        capture = self._capture(sequence)
        fps = capture.get(cv2.CAP_PROP_FPS)
        items = []
        for position in positions:
            record = records[position]
            item = self._load_frame(sequence, record)
            item["rgb"] = self._read_rgb(sequence, record["frame_id"])
            item["frame_ids"] = torch.tensor(record["frame_id"], dtype=torch.long)
            item["timestamps"] = torch.tensor(record["frame_id"] / fps, dtype=torch.float64)
            items.append(item)

        sample = {key: torch.stack([item[key] for item in items]) for key in items[0]}
        if self.mode == "frame":
            sample = {key: value[0] for key, value in sample.items()}
        sample["scene"] = sequence.parent.name
        sample["variant"] = sequence.name
        return sample

    def close(self):
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
        self._capture_positions.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_captures"] = {}
        state["_capture_positions"] = {}
        state["_capture_pid"] = None
        return state

    def __del__(self):
        if hasattr(self, "_captures"):
            self.close()

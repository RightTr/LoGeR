"""MovingDrone video frames and CSV camera centers."""

import csv

import torch

from .base import BaseSequenceDataset


class MovingDroneDataset(BaseSequenceDataset):
    def __init__(self, root="/mnt/Data/RIght/MovingDrone_Long", variant="baseline", **kwargs):
        self.variant = variant
        super().__init__(root, **kwargs)

    def _discover_sequences(self):
        return sorted(
            path for path in self.root.glob(f"*/{self.variant}")
            if (path / "video.mp4").is_file() and (path / "poses.csv").is_file()
        )

    def _load_records(self, sequence):
        with (sequence / "poses.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        records = [
            {
                "frame_id": int(row["frame_id"]),
                "position_xyz": torch.tensor([float(row[k]) for k in ("x", "y", "z")]),
            }
            for row in rows
        ]
        records.sort(key=lambda row: row["frame_id"])
        return records

    def _load_frame(self, sequence, record):
        return {"position_xyz": record["position_xyz"]}

"""Small, self-contained checks for MovingDrone's confirmed RGB/CSV layout."""

import csv
import pickle
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from loger.datasets import MovingDroneDataset


class MovingDroneDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ("sequence_01", "sequence_02"):
            sequence = self.root / name / "baseline"
            sequence.mkdir(parents=True)
            writer = cv2.VideoWriter(str(sequence / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (16, 16))
            self.assertTrue(writer.isOpened())
            for frame_id in range(8):
                writer.write(np.full((16, 16, 3), frame_id * 25, dtype=np.uint8))
            writer.release()
            with (sequence / "poses.csv").open("w", newline="") as file:
                rows = csv.DictWriter(file, fieldnames=("frame_id", "x", "y", "z"))
                rows.writeheader()
                for frame_id in (0, 2, 4, 6):
                    rows.writerow({"frame_id": frame_id, "x": frame_id, "y": 1, "z": 2})

    def tearDown(self):
        self.temp.cleanup()

    def test_windows_use_video_frame_ids_and_stay_within_scene(self):
        dataset = MovingDroneDataset(self.root, window_size=2, stride=1, image_size=(8, 8))
        self.assertEqual(len(dataset), 6)
        sample = dataset[1]
        self.assertEqual(sample["frame_ids"].tolist(), [2, 4])
        self.assertEqual(sample["rgb"].shape, (2, 3, 8, 8))
        self.assertEqual(sample["position_xyz"][:, 0].tolist(), [2, 4])
        self.assertAlmostEqual(float(sample["rgb"][0].mean()), 50 / 255, delta=0.03)
        self.assertEqual(dataset[3]["scene"], "sequence_02")
        dataset.close()

    def test_frame_step_range_and_worker_handles(self):
        dataset = MovingDroneDataset(
            self.root, scenes=("sequence_01",), window_size=2, frame_step=2,
            frame_range=(2, 7), image_size=(8, 8),
        )
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["frame_ids"].tolist(), [2, 6])
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(restored[0]["frame_ids"].tolist(), [2, 6])
        dataset.close()
        for batch in DataLoader(dataset, batch_size=1, num_workers=2):
            self.assertEqual(batch["frame_ids"].tolist(), [[2, 6]])
        dataset.close()
        restored.close()

    def test_frame_mode(self):
        dataset = MovingDroneDataset(self.root, mode="frame", scenes=("sequence_01",))
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset[2]["frame_ids"].item(), 4)
        self.assertEqual(dataset[2]["rgb"].shape, (3, 280, 504))
        dataset.close()


if __name__ == "__main__":
    unittest.main()

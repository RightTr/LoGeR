#!/usr/bin/env python3
"""Export official RoMa GP features and distill them into LoGeR projections.

MegaDepth layout follows RoMa/DKM: ROOT/prep_scene_info/*.npy and the image
paths stored in each scene-info file, relative to ROOT. Both modes use the
same deterministic resize and normalization; no depth or pose is required.

Export (official RoMa Outdoor and non-Reg DINOv2-L checkpoints):
  python scripts/distill_roma.py export-gp --megadepth ROOT --gp-dir GP_DIR \
      --roma-checkpoint roma_outdoor.pth --teacher-dino dinov2_vitl14_pretrain.pth

Train the UAV and target projections using those exported GP features:
  python scripts/distill_roma.py distill --megadepth ROOT --gp-dir GP_DIR \
      --roma-checkpoint roma_outdoor.pth --loger-checkpoint loger.pt \
      --satellite-dino dinov2_vitl14_reg_pretrain.pth --output calibrated.pt

Use the same --image-size and --min-overlap in both modes. A LoGeR checkpoint
containing satellite_encoder.backbone weights can replace --satellite-dino.
The output has a roma_encoder state dictionary compatible with the geometry
stage in train/train_roma_adapter.py.
"""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loger.models.roma import RoMaCoarseEncoder
from loger.models.satellite_encoder import SatelliteEncoder
from loger.models.uav_encoder import UAVEncoder
from roma.teacher import RoMaGPTeacher


class MegaDepthPairs(Dataset):
    # Official RoMa train_loftr split and its ignored benchmark scenes.
    EXCLUDED = {"0015", "0022", "0121", "0133", "0168", "0178", "0229", "0349",
                "0412", "0430", "0443", "1001", "5014", "5015", "5016",
                "0008", "0019", "0021", "0024", "0025", "0032", "0063", "1589"}

    def __init__(self, root, image_size, min_overlap, max_pairs=None):
        self.root = Path(root)
        self.image_size = image_size
        self.pairs = []
        for scene_path in sorted((self.root / "prep_scene_info").glob("*.npy")):
            if scene_path.stem in self.EXCLUDED:
                continue
            scene = np.load(scene_path, allow_pickle=True).item()
            for (i, j), overlap in zip(scene["pairs"], scene["overlaps"]):
                if min_overlap < overlap < 1.0:
                    self.pairs.append((str(scene["image_paths"][i]), str(scene["image_paths"][j])))
                    if max_pairs and len(self.pairs) >= max_pairs:
                        return
        if not self.pairs:
            raise ValueError(f"No MegaDepth image pairs in {self.root / 'prep_scene_info'}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        name_a, name_b = self.pairs[index]
        def load(name):
            with Image.open(self.root / name) as image:
                image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
                value = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
            mean = value.new_tensor((0.485, 0.456, 0.406))[:, None, None]
            std = value.new_tensor((0.229, 0.224, 0.225))[:, None, None]
            return (value - mean) / std
        identity = f"{self.image_size}\0{name_a}\0{name_b}".encode()
        key = hashlib.sha256(identity).hexdigest()
        return {"image_a": load(name_a), "image_b": load(name_b), "key": key,
                "name_a": name_a, "name_b": name_b}


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("model_state_dict", state.get("state_dict", state))
    return {name.removeprefix("module."): value for name, value in state.items()}


def student_encoders(loger_path, satellite_path, device):
    state = UAVEncoder.remap_state_dict(load_checkpoint(loger_path))
    uav = UAVEncoder(pretrained=False)
    prefix = "uav_encoder."
    uav.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                         if key.startswith(prefix)}, strict=True)
    satellite = SatelliteEncoder(pretrained=False)
    if satellite_path:
        satellite_state = load_checkpoint(satellite_path)
        prefix = "satellite_encoder.backbone."
        selected = {key[len(prefix):]: value for key, value in satellite_state.items()
                    if key.startswith(prefix)}
        satellite.backbone.load_state_dict(selected or satellite_state, strict=True)
    else:
        prefix = "satellite_encoder."
        satellite.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                   if key.startswith(prefix)}, strict=True)
    return uav.requires_grad_(False).eval().to(device), satellite.requires_grad_(False).eval().to(device)


def run_export(args, loader, device):
    teacher = RoMaGPTeacher(args.teacher_dino, args.roma_checkpoint).to(device).eval()
    output = Path(args.gp_dir)
    output.mkdir(parents=True, exist_ok=True)
    for step, batch in enumerate(loader, 1):
        gp = teacher(batch["image_a"].to(device), batch["image_b"].to(device)).cpu()
        for index, key in enumerate(batch["key"]):
            torch.save({"gp_posterior": gp[index], "image_a": batch["name_a"][index],
                        "image_b": batch["name_b"][index], "image_size": args.image_size},
                       output / f"{key}.pt")
        print(f"exported batch {step}/{len(loader)}", flush=True)


def run_distill(args, loader, device):
    uav, satellite = student_encoders(args.loger_checkpoint, args.satellite_dino, device)
    model = RoMaCoarseEncoder(args.roma_checkpoint).to(device).train()
    # Both input streams start from RoMa's shared projection, then train independently.
    model.satellite_proj.load_state_dict(model.proj.state_dict())
    optimizer = torch.optim.AdamW(list(model.proj.parameters()) + list(model.satellite_proj.parameters()),
                                  lr=args.learning_rate)
    grid = (args.image_size // 14, args.image_size // 14)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader, 1):
            image_a, image_b = batch["image_a"].to(device), batch["image_b"].to(device)
            teacher_gp = torch.stack([torch.load(Path(args.gp_dir) / f"{key}.pt", map_location="cpu",
                                                 weights_only=True)["gp_posterior"]
                                      for key in batch["key"]]).to(device)
            with torch.no_grad():
                f1 = uav(image_a, is_training=True)["x_norm_patchtokens"]
                f2 = satellite(image_b)["patch_tokens"]
            predicted = model(f1, f2, grid, grid)["gp_posterior"]
            loss = F.mse_loss(F.normalize(predicted, dim=1), F.normalize(teacher_gp, dim=1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % args.log_every == 0 or step == len(loader):
                print(f"epoch {epoch + 1}/{args.epochs} batch {step}/{len(loader)} loss={loss.item():.6f}", flush=True)
        torch.save({"roma_encoder": model.state_dict(), "epoch": epoch + 1}, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("export-gp", "distill"))
    parser.add_argument("--megadepth", required=True)
    parser.add_argument("--gp-dir", required=True)
    parser.add_argument("--roma-checkpoint", required=True)
    parser.add_argument("--teacher-dino")
    parser.add_argument("--loger-checkpoint")
    parser.add_argument("--satellite-dino")
    parser.add_argument("--output")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--min-overlap", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.image_size % 14:
        parser.error("--image-size must be divisible by 14")
    if args.mode == "export-gp" and not args.teacher_dino:
        parser.error("export-gp requires --teacher-dino")
    if args.mode == "distill" and (not args.loger_checkpoint or not args.output):
        parser.error("distill requires --loger-checkpoint and --output")
    torch.set_float32_matmul_precision("highest")
    dataset = MegaDepthPairs(args.megadepth, args.image_size, args.min_overlap, args.max_pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.mode == "distill",
                        num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    if args.mode == "export-gp":
        run_export(args, loader, torch.device(args.device))
    else:
        run_distill(args, loader, torch.device(args.device))


if __name__ == "__main__":
    main()

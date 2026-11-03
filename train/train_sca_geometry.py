"""Train only SCA through frozen LoGeR geometry outputs."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loger.models.pi3 import Pi3
from loger.utils.satellite import SatelliteTileManager
from train.loss import LoGeRLoss, rotation_geodesic
from train.scene_adapter import iter_chunks


def align_centers(predicted, target):
    source_mean = predicted.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    source = predicted - source_mean
    destination = target - target_mean
    u, _, vh = torch.linalg.svd(source.transpose(-1, -2) @ destination)
    sign = torch.linalg.det(u @ vh)
    diagonal = torch.diag_embed(torch.stack((torch.ones_like(sign), torch.ones_like(sign), sign), -1))
    rotation = u @ diagonal @ vh
    scale = (source @ rotation * destination).sum(dim=(1, 2), keepdim=True)
    scale = scale / source.square().sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return source @ rotation * scale + target_mean


def load_model(config, device):
    with open(config["model_config"]) as source:
        options = yaml.safe_load(source)["model"]
    options["sca_enabled"] = True
    model = Pi3(**{key: value for key, value in options.items()
                   if key in Pi3.__init__.__code__.co_varnames})
    checkpoint = torch.load(config["weights"], map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.to(device).eval()


def memory_for_scene(model, entry):
    with SatelliteTileManager(entry["dop"]) as manager:
        tiles = manager.get_all_tiles()
    with torch.no_grad():
        return model.encode_satellite(tiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sca_geometry.yaml")
    args = parser.parse_args()
    with open(args.config) as source:
        config = yaml.safe_load(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    optimizer = torch.optim.AdamW(model.sca.parameters(), lr=config["learning_rate"])
    criterion = LoGeRLoss(
        lambda_global=config.get("lambda_global", 1.0),
        lambda_rotation=config.get("lambda_rotation", 0.1),
        lambda_translation=config.get("lambda_translation", 10.0),
    )
    scenes = {phase: [(entry, memory_for_scene(model, entry)) for entry in config[phase]]
              for phase in ("train", "val")}
    window, overlap = config["window_size"], config["overlap_size"]
    length = 2 * window - overlap
    with SummaryWriter(config["log_dir"]) as writer:
        for epoch in range(config["epochs"]):
            criterion.lambda_global = config.get("lambda_global", 1.0) if epoch >= config["warmup_epochs"] else 0.0
            for phase in ("train", "val"):
                values = {name: [] for name in ("total", "local", "pose", "rotation", "translation", "global", "sequence_scale")}
                ate, rpe_translation, rpe_rotation = [], [], []
                for entry, memory in scenes[phase]:
                    for batch in iter_chunks(entry, length):
                        images = batch["images"].to(device)
                        poses = batch["poses"].to(device)
                        depth = batch["depth"].to(device)
                        intrinsics = batch["intrinsics"].to(device)
                        targets = {"poses": poses, "depth": depth, "intrinsics": intrinsics}
                        if "valid_mask" in batch:
                            targets["valid_mask"] = batch["valid_mask"].to(device)
                        with torch.set_grad_enabled(phase == "train"):
                            output = model(images, satellite_features=memory, window_size=window,
                                           overlap_size=overlap, no_detach=True)
                            terms = criterion(output, targets)
                            total = terms["total"]
                        if phase == "train":
                            optimizer.zero_grad()
                            total.backward()
                            grad = torch.nn.utils.clip_grad_norm_(model.sca.parameters(), 1.0)
                            optimizer.step()
                            writer.add_scalar("sca/grad_norm", grad.item(), epoch)
                            writer.add_scalar("sca/gate", model.sca.gate.item(), epoch)
                        for name, value in terms.items():
                            values[name].append(value.detach().float().mean().item())
                        if phase == "val":
                            with torch.no_grad():
                                stitched = model(images, satellite_features=memory, window_size=window,
                                                 overlap_size=overlap, sim3=True, no_detach=True)
                            predicted = stitched["camera_poses"][..., :3, 3]
                            target = poses[..., :3, 3]
                            aligned = align_centers(predicted, target)
                            ate.append((aligned - target).square().sum(-1).mean().sqrt().item())
                            rpe_translation.append((torch.diff(aligned, dim=1) - torch.diff(target, dim=1))
                                                   .square().sum(-1).mean().sqrt().item())
                            pred_rel = (torch.linalg.inv(stitched["camera_poses"][:, :-1])
                                        @ stitched["camera_poses"][:, 1:])
                            gt_rel = torch.linalg.inv(poses[:, :-1]) @ poses[:, 1:]
                            rpe_rotation.append(rotation_geodesic(pred_rel[..., :3, :3],
                                                                  gt_rel[..., :3, :3]).mean().item())
                for name, scores in values.items():
                    if scores:
                        tag = f"Loss/{name}" if phase == "train" else f"Validation/loss_{name}"
                        writer.add_scalar(tag, sum(scores) / len(scores), epoch)
                if ate:
                    writer.add_scalar("Validation/ate_rmse", sum(ate) / len(ate), epoch)
                    writer.add_scalar("Validation/rpe_translation", sum(rpe_translation) / len(rpe_translation), epoch)
                    writer.add_scalar("Validation/rpe_rotation", sum(rpe_rotation) / len(rpe_rotation), epoch)
            torch.save(model.sca.state_dict(), Path(config["log_dir"]) / "sca_geometry.pt")


if __name__ == "__main__":
    main()

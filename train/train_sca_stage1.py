"""Stage 1: learn UAV-to-DOP occupancy without changing LoGeR geometry."""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loger.models.pi3 import Pi3
from loger.utils.satellite import SatelliteTileManager
from train.satellite_supervision import project, occupancy, kl
from train.scene_adapter import iter_chunks
from tools.visualize_sca import dop_image, overlay, trajectory, mark


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


def prepare_scene(model, entry):
    with SatelliteTileManager(entry["dop"]) as manager:
        tiles = manager.get_all_tiles()
    with torch.no_grad():
        memory = model.encode_satellite(tiles)
    with rasterio.open(entry["dop"]) as source:
        size = source.width, source.height
    return entry, memory, size


def metrics(predicted, target, valid, coords, centers, center_valid):
    support = target > 0
    count = support.sum(-1).clamp_min(1)
    selected = predicted.topk(int(count.max().item()), dim=-1).indices
    rank = torch.arange(selected.shape[-1], device=selected.device)
    chosen = torch.zeros_like(support).scatter(-1, selected,
                                               (rank < count[..., None]).expand_as(selected))
    intersection = (chosen & support).sum(-1).float()
    union = (chosen | support).sum(-1).clamp_min(1)
    iou = (intersection / union)[valid].mean()
    peak = coords[predicted.argmax(-1)]
    error = (peak - centers).norm(dim=-1)[valid & center_valid]
    return error.mean().item() if error.numel() else float("nan"), iou.item(), peak


def scene_epoch(model, scene, config, device, optimizer=None):
    entry, memory, size = scene
    coordinates = memory["pool_map_coords"].to(device)[0]
    losses, frame_losses, errors, ious = [], [], [], []
    tracks_gt, tracks_pred = [], []
    sample = None
    for batch in iter_chunks(entry, config["window_size"]):
        images, depth, intrinsics, poses, transform = (batch[name].to(device) for name in
            ("images", "depth", "intrinsics", "poses", "world_to_dop"))
        output = model(images, satellite_features=memory, window_size=config["window_size"], no_detach=True)
        frame_pred = output["satellite_attention"]["frame"]
        chunk_pred = output["satellite_attention"]["chunk"]
        points, valid = project(depth, intrinsics, poses, transform)
        frame_gt, chunk_gt, frame_valid = occupancy(points, valid, memory["tile_origins"],
                                                    memory["pool_valid_mask"], *size)
        if not frame_valid.any():
            continue
        frame_loss = kl(frame_gt, frame_pred, frame_valid)
        chunk_loss = kl(chunk_gt, chunk_pred, frame_valid.any(dim=1))
        loss = frame_loss + config["chunk_weight"] * chunk_loss
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        frame_losses.append(frame_loss.item())
        center, center_valid = project(depth, intrinsics, poses, transform, sample_size=1)
        center, center_valid = center[..., 0, :], center_valid[..., 0]
        error, iou, peak = metrics(frame_pred, frame_gt, frame_valid, coordinates,
                                   center, center_valid)
        errors.append(error)
        ious.append(iou)
        track_mask = (frame_valid & center_valid)[0]
        tracks_gt.extend(center[0, track_mask].detach().cpu().tolist())
        tracks_pred.extend(peak[0, track_mask].detach().cpu().tolist())
        if sample is None:
            first = frame_valid[0].nonzero()[0, 0].item()
            gt_point = center[0, first] if center_valid[0, first] else points[0, first, valid[0, first]].mean(dim=0)
            sample = (frame_gt[0, first].detach().cpu().numpy(),
                      frame_pred[0, first].detach().cpu().numpy(),
                      chunk_gt[0].detach().cpu().numpy(),
                      chunk_pred[0].detach().cpu().numpy(),
                      gt_point.detach().cpu().tolist(),
                      peak[0, first].detach().cpu().tolist())
    return np.mean(losses), np.mean(frame_losses), np.nanmean(errors), np.mean(ious), sample, tracks_gt, tracks_pred


def write_images(writer, entry, memory, result, epoch):
    _, _, _, _, sample, gt_track, pred_track = result
    if sample is None:
        return
    image, size = dop_image(entry["dop"])
    coords = memory["pool_map_coords"][0].cpu().numpy()
    names = ("frame_gt", "frame_pred", "chunk_gt", "chunk_pred")
    for name, weights in zip(names, sample[:4]):
        panel = overlay(image, size, weights, coords)
        if name.startswith("frame"):
            panel = mark(panel, size, sample[4], sample[5])
        writer.add_image(f"val/{name}", panel, epoch, dataformats="HWC")
    if gt_track:
        background = overlay(image, size, sample[3], coords)
        writer.add_image("val/trajectory", trajectory(background, size, np.asarray(gt_track),
                         np.asarray(pred_track)), epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sca_stage1.yaml")
    parser.add_argument("--log-dir")
    args = parser.parse_args()
    with open(args.config) as source:
        config = yaml.safe_load(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    train = [prepare_scene(model, entry) for entry in config["train"]]
    val = [prepare_scene(model, entry) for entry in config["val"]]
    optimizer = torch.optim.AdamW(model.sca.parameters(), lr=config["learning_rate"])
    log_dir = Path(args.log_dir or config["log_dir"])
    with SummaryWriter(str(log_dir)) as writer:
        for epoch in range(config["epochs"]):
            for scene in train:
                result = scene_epoch(model, scene, config, device, optimizer)
                writer.add_scalar("train/loss_total", result[0], epoch)
                writer.add_scalar("train/loss_frame", result[1], epoch)
            with torch.no_grad():
                for scene in val:
                    result = scene_epoch(model, scene, config, device)
                    writer.add_scalar("val/loss_total", result[0], epoch)
                    writer.add_scalar("val/mean_pixel_error", result[2], epoch)
                    writer.add_scalar("val/heatmap_iou", result[3], epoch)
                    write_images(writer, scene[0], scene[1], result, epoch)
            torch.save(model.sca.state_dict(), log_dir / "sca_stage1.pt")


if __name__ == "__main__":
    main()

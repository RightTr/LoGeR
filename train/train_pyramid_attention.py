"""Train independent L0/L1/L2 satellite attention with pixel labels."""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loger.models.pi3 import Pi3
from loger.utils.satellite_pyramid import read_pyramid
from tools.visualize_pyramid_attention import render


def sequences(scene, labels, length, step):
    with (Path(scene) / "poses.csv").open() as source:
        rows = list(csv.DictReader(source))
    with open(labels) as source:
        locations = {int(row["frame_id"]): (float(row["x"]), float(row["y"]))
                     for row in csv.DictReader(source)}
    video = cv2.VideoCapture(str(Path(scene) / "video.mp4"))
    for start in range(0, len(rows) - length + 1, step):
        chosen = rows[start:start + length]
        ids = [int(row["frame_id"]) for row in chosen]
        if any(frame_id not in locations for frame_id in ids):
            continue
        frames = []
        for frame_id in ids:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = video.read()
            if not ok:
                break
            frame = cv2.cvtColor(cv2.resize(frame, (504, 280)), cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255)
        if len(frames) == length:
            yield torch.stack(frames)[None], torch.tensor([locations[i] for i in ids])[None]
    video.release()


def coverage(gt, raw):
    result = torch.zeros(gt.shape[:-1], dtype=torch.bool)
    for (left, top), valid in zip(raw["local"]["tile_origins"], raw["local"]["pixel_valid_mask"]):
        x = (gt[..., 0] - left).long()
        y = (gt[..., 1] - top).long()
        inside = (x >= 0) & (x < 518) & (y >= 0) & (y < 518)
        result |= inside & valid[y.clamp(0, 517), x.clamp(0, 517)]
    return result


def level_metrics(weights, gt, level, covered, sigma):
    coords = level["coords"].to(weights.device).float()
    mask = level["mask"].to(weights.device)
    distance = torch.cdist(gt.float(), coords.expand(gt.shape[0], -1, -1))
    target_logits = (-0.5 * (distance / sigma).square()).masked_fill(~mask[:, None], -torch.inf)
    target = target_logits.softmax(dim=-1)
    valid = covered.to(weights.device)
    loss = (target * (target.clamp_min(1e-12).log() - weights.clamp_min(1e-12).log())).sum(-1)
    loss = loss[valid].mean() if valid.any() else weights.sum() * 0
    selected = coords.expand(gt.shape[0], -1, -1).gather(
        1, weights.argmax(-1)[..., None].expand(-1, -1, 2))
    error = (selected - gt).norm(dim=-1)[valid]
    mass = weights.masked_fill(distance > 64, 0).sum(-1)[valid]
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    entropy = (entropy / mask.sum(-1).float().log().clamp_min(1)[:, None])[valid]
    return loss, {
        "top1_error_px": error.mean().item() if error.numel() else float("nan"),
        "recall_64px": (error <= 64).float().mean().item() if error.numel() else float("nan"),
        "gt_mass": mass.mean().item() if mass.numel() else float("nan"),
        "entropy": entropy.mean().item() if entropy.numel() else float("nan"),
        "coverage": valid.float().mean().item(),
    }


def load_model(config, device):
    with open(config["model_config"]) as source:
        options = yaml.safe_load(source)["model"]
    options["pyramid_attention_enabled"] = True
    model = Pi3(**{key: value for key, value in options.items()
                   if key in Pi3.__init__.__code__.co_varnames})
    checkpoint = torch.load(config["weights"], map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.to(device).eval()


def prepare_scenes(model, entries):
    scenes = []
    for entry in entries:
        raw = read_pyramid(entry["dop"], entry["center"])
        with torch.no_grad():
            memory = model.encode_pyramid(raw)
        scenes.append((entry, raw, memory))
    return scenes


def run_scene(model, entry, raw, memory, config, writer, phase, step, optimizer=None, log_image=False):
    window, overlap = config["window_size"], config["overlap_size"]
    length = 2 * window - overlap
    summaries = {name: [] for name in ("L0", "L1", "L2")}
    first_image = None
    for images, gt in sequences(entry["scene"], entry["labels"], length, window - overlap):
        images, gt = images.to(next(model.parameters()).device), gt.to(next(model.parameters()).device)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(images, pyramid_features=memory, window_size=window,
                           overlap_size=overlap, no_detach=True)
            maps = output["pyramid_attention"]
        local_covered = coverage(gt.cpu(), raw).to(gt.device)
        losses = []
        for name in ("L0", "L1", "L2"):
            valid = ((gt[..., 0] >= 0) & (gt[..., 0] < raw["width"]) &
                     (gt[..., 1] >= 0) & (gt[..., 1] < raw["height"])) if name == "L0" else local_covered
            loss, metrics = level_metrics(maps[name], gt, memory[name], valid, config["sigma_px"])
            summaries[name].append(metrics)
            if valid.any():
                losses.append(loss)
        total = torch.stack(losses).mean()
        if optimizer is not None:
            optimizer.zero_grad()
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.pyramid_attention.parameters(), 1.0)
            optimizer.step()
            writer.add_scalar("train/loss", total.item(), step)
            writer.add_scalar("train/grad_norm", grad.item(), step)
            for name in ("L0", "L1", "L2"):
                writer.add_scalar(f"train/temperature/{name}",
                                  model.pyramid_attention.log_temperature[name].exp().item(), step)
        if first_image is None and log_image:
            first_image = (images[0, 0].cpu(), gt[0, 0].cpu().tolist(),
                           {name: maps[name][0, 0].detach().cpu().numpy() for name in maps})
        step += 1
    for name, rows in summaries.items():
        for metric in ("top1_error_px", "recall_64px", "gt_mass", "entropy", "coverage"):
            values = [row[metric] for row in rows if np.isfinite(row[metric])]
            if values:
                writer.add_scalar(f"{phase}/{metric}/{name}", np.mean(values), step)
    if first_image is not None:
        image, location, attention = first_image
        mosaic = render(entry["dop"], image, memory, attention, location,
                        raw["local"]["tile_origins"].tolist())
        writer.add_image(f"{phase}/pyramid/{Path(entry['scene']).name}", mosaic, step)
    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pyramid_attention.yaml")
    parser.add_argument("--log-dir")
    args = parser.parse_args()
    with open(args.config) as source:
        config = yaml.safe_load(source)
    log_dir = args.log_dir or config["log_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    train = prepare_scenes(model, config["train"])
    val = prepare_scenes(model, config["val"])
    optimizer = torch.optim.AdamW(model.pyramid_attention.parameters(), lr=config["learning_rate"])
    with SummaryWriter(log_dir) as writer:
        step, last_image = 0, -config["image_every_steps"]
        for entry, raw, memory in val:
            run_scene(model, entry, raw, memory, config, writer, "val", step, log_image=True)
        for epoch in range(config["epochs"]):
            for entry, raw, memory in train:
                step = run_scene(model, entry, raw, memory, config, writer, "train", step, optimizer)
            visualize = step - last_image >= config["image_every_steps"]
            for entry, raw, memory in val:
                step = run_scene(model, entry, raw, memory, config, writer, "val", step,
                                 log_image=visualize)
            if visualize:
                last_image = step
            torch.save(model.pyramid_attention.state_dict(), Path(log_dir) / "pyramid_attention.pt")


if __name__ == "__main__":
    main()

"""Compare trained Satellite Memory variants on MovingDrone_Long sequences."""

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.datasets_preprocess.movingdrone import load_frames
from eval.uav_satelitte.run_movingdrone import sim3
from train.train_satellite_memory import build_model


def tensor_to(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: tensor_to(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tensor_to(item, device) for item in value)
    return value


def load_dop(sequence):
    cropped = sequence / "dop_view.jpg"
    if not cropped.is_file():
        cropped = next((sequence / name for name in
                        ("dop/2023_cropped.jpg", "dop/2025_cropped.jpg")
                        if (sequence / name).is_file()), sequence / "dop.jpg")
    if cropped.is_file():
        dop_path = cropped
    else:
        meta = json.loads((sequence / "meta.json").read_text())
        dops = meta["dops"]
        dop = dops.get("dop_2023", next(iter(dops.values())))
        dop_path = sequence / dop["file"]
    image = Image.open(dop_path).convert("RGB")
    width, height = image.size
    scale = 518 / max(width, height)
    out_h = max(14, round(height * scale / 14) * 14)
    out_w = max(14, round(width * scale / 14) * 14)
    image = image.resize((out_w, out_h), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)


def load_contiguous_frames(sequence, count, start_index, size=(448, 252)):
    with (sequence / "poses.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    rows = rows[start_index:start_index + count]
    video = cv2.VideoCapture(str(sequence / "video.mp4"))
    images, gt, frame_ids = [], [], []
    previous_id = None
    for row in rows:
        frame_id = int(row["frame_id"])
        if previous_id is None or frame_id != previous_id + 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = video.read()
        if not ok:
            break
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div_(255))
        gt.append([float(row[key]) for key in ("x", "y", "z")])
        frame_ids.append(frame_id)
        previous_id = frame_id
    video.release()
    return torch.stack(images), np.asarray(gt), np.asarray(frame_ids)


def run_sequence(model, sequence, label, device, output, count, start_index=None,
                full_trajectory=False):
    if full_trajectory:
        with (sequence / "poses.csv").open(newline="") as file:
            count = sum(1 for _ in csv.DictReader(file))
        images, gt, frame_ids = load_contiguous_frames(
            sequence, count, 0, size=(448, 252))
    elif start_index is None:
        images, gt, frame_ids = load_frames(sequence, count, size=(448, 252))
    else:
        images, gt, frame_ids = load_contiguous_frames(sequence, count, start_index,
                                                       size=(448, 252))
    images = images[None] if full_trajectory else images[None].to(device)
    tile = load_dop(sequence)[None].to(device)
    print(f"{label}: satellite image={next((sequence / name for name in ('dop_view.jpg', 'dop/2023_cropped.jpg', 'dop/2025_cropped.jpg', 'dop.jpg') if (sequence / name).is_file()), 'meta DOP')}", flush=True)
    with torch.no_grad():
        sat_features = model.encode_roma_satellite(tile)
        normalized = (tile - model.image_mean) / model.image_std
        roma_target = model.roma_matcher.encode_target(normalized)

    modes = ("off", "reset", "recurrent")
    labels = {"off": "LoGeR", "reset": "Probabilistic, no memory",
              "recurrent": "Probabilistic, recurrent memory"}
    result = {"frame_ids": frame_ids.tolist(), "modes": {}}
    trajectories = {}
    for mode in modes:
        if full_trajectory:
            chunk_size, segment_overlap = 64, 8
            starts = list(range(0, max(1, count - chunk_size + 1),
                                chunk_size - segment_overlap))
            last = max(0, count - chunk_size)
            if starts[-1] != last:
                starts.append(last)
            pieces = []
            memory = None
            previous_end = 0
            for start in starts:
                end = min(count, start + chunk_size)
                chunk = images[:, start:end].to(device)
                fresh = (torch.arange(start, end, device=device) >= previous_end)[None]
                with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prediction = model(
                        chunk,
                        roma_tile_image=tile,
                        roma_tile_id=torch.tensor([0], device=device),
                        roma_tile_bbox=torch.tensor([[0., 0., 1., 1.]], device=device),
                        roma_satellite_features=tensor_to(sat_features, device),
                        roma_match_target_features=tensor_to(roma_target, device),
                        roma_memory=memory if mode in ("recurrent", "shuffle") else None,
                        roma_new_frame_mask=fresh,
                        roma_memory_mode=mode,
                        window_size=6,
                        overlap_size=2,
                        num_iterations=1,
                        no_detach=True,
                    )
                centers_part = prediction["camera_poses"][0, :, :3, 3].float().cpu().numpy().astype(np.float64)
                overlap = max(0, previous_end - start)
                if pieces and overlap:
                    joined = np.concatenate(pieces, axis=0)
                    scale_seg, rotation_seg, translation_seg = sim3(
                        centers_part[:overlap], joined[-overlap:])
                    centers_part = scale_seg * (centers_part @ rotation_seg.T) + translation_seg
                pieces.append(centers_part[overlap:])
                if mode in ("recurrent", "shuffle"):
                    memory = prediction["satellite_memory"]
                previous_end = end
                del prediction, chunk
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"{label} {labels[mode]}: frames {start}-{end - 1}/{count - 1}",
                      flush=True)
            centers = np.concatenate(pieces, axis=0)
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                       enabled=device.type == "cuda"):
                prediction = model(
                    images,
                    roma_tile_image=tile,
                    roma_tile_id=torch.tensor([0], device=device),
                    roma_tile_bbox=torch.tensor([[0., 0., 1., 1.]], device=device),
                    roma_satellite_features=tensor_to(sat_features, device),
                    roma_match_target_features=tensor_to(roma_target, device),
                    roma_memory_mode=mode,
                    window_size=6,
                    overlap_size=2,
                    num_iterations=1,
                    no_detach=True,
                )
            centers = prediction["camera_poses"][0, :, :3, 3].float().cpu().numpy().astype(np.float64)
            del prediction
        scale, rotation, translation = sim3(centers, gt)
        aligned = scale * (centers @ rotation.T) + translation
        errors = np.linalg.norm(aligned - gt, axis=1)
        relative = np.linalg.norm(np.diff(aligned, axis=0) - np.diff(gt, axis=0), axis=1)
        result["modes"][mode] = {
            "ate_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "ate_mean_m": float(errors.mean()),
            "relative_translation_rmse_m": float(np.sqrt(np.mean(relative ** 2))),
        }
        trajectories[mode] = aligned
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(label, labels[mode], result["modes"][mode], flush=True)

    output.mkdir(parents=True, exist_ok=True)
    (output / f"{label}.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=2.2, label="GT")
    colors = {"off": "#4c78a8", "reset": "#f58518", "recurrent": "#54a24b"}
    for mode in modes:
        ate = result["modes"][mode]["ate_rmse_m"]
        ax.plot(trajectories[mode][:, 0], trajectories[mode][:, 1],
                color=colors[mode], linewidth=1.8, label=f"{labels[mode]} ({ate:.1f} m)")
    ax.set_title(f"{label} | {count} {'consecutive' if full_trajectory else 'sampled'} frames")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.axis("equal"); ax.grid(alpha=.25); ax.legend()
    fig.savefig(output / f"{label}.png", dpi=180)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/satellite_memory_tiles.yaml")
    parser.add_argument("--root", type=Path, default=Path("/mnt/Data/RIght/MovingDrone_Long"))
    parser.add_argument("--scenes", nargs="+", default=["airport10", "olympiastadion1", "rural1"])
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--contiguous-clips", type=int, default=0,
                        help="Evaluate contiguous clips sampled across each sequence; 0 uses uniform sparse sampling")
    parser.add_argument("--full-trajectory", action="store_true",
                        help="Evaluate every video frame in chronological order")
    parser.add_argument("--output", type=Path, default=Path("runs/satellite_memory_long"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device or config["device"])
    model = build_model(config, device).eval()
    checkpoint = args.checkpoint or Path(config["output"]) / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.satellite_probabilistic.load_state_dict(state["satellite_probabilistic"])

    all_results = {}
    for name in args.scenes:
        sequence = args.root / name / args.variant if args.variant else args.root / name
        label = name.replace("/", "_")
        if args.full_trajectory:
            result = run_sequence(model, sequence, label, device, args.output,
                                  args.frames, full_trajectory=True)
            all_results[label] = result["modes"]
        elif args.contiguous_clips:
            with (sequence / "poses.csv").open(newline="") as file:
                num_records = sum(1 for _ in csv.DictReader(file))
            last_start = max(0, num_records - args.frames)
            starts = np.linspace(0, last_start, max(1, args.contiguous_clips), dtype=int)
            clips = []
            for index, start in enumerate(starts):
                result = run_sequence(model, sequence, f"{label}_clip{index + 1:02d}",
                                      device, args.output, args.frames,
                                      start_index=int(start))
                clips.append({"start_frame": int(start), "metrics": result["modes"]})
            averages = {
                mode: {key: float(np.mean([clip["metrics"][mode][key] for clip in clips]))
                       for key in clips[0]["metrics"][mode]}
                for mode in clips[0]["metrics"]
            }
            all_results[label] = {"mean_over_contiguous_clips": averages, "clips": clips}
            print(label, "mean over contiguous clips", averages, flush=True)
        else:
            result = run_sequence(model, sequence, label, device, args.output, args.frames)
            all_results[label] = result["modes"]
    (args.output / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"Saved Long evaluation to {args.output}", flush=True)


if __name__ == "__main__":
    main()

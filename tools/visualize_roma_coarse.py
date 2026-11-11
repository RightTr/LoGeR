"""Visualize MatchAnything RoMa coarse matches from a UAV frame to its DOP."""

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roma.coarse_matcher import RoMaCoarseMatcher


def resize_dop(path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = 518 / max(width, height)
    out_h = max(14, round(height * scale / 14) * 14)
    out_w = max(14, round(width * scale / 14) * 14)
    image = image.resize((out_w, out_h), Image.Resampling.BICUBIC)
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
    return array, tensor


def load_frame(video_path, frame_id, size=(448, 252)):
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_id}: {video_path}")
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path,
                        default=Path("/mnt/Data/RIght/MovingDrone_Tiles/airport10/tile01"))
    parser.add_argument("--config", type=Path, default=Path("configs/roma_hidden_adapter_tiles.yaml"))
    parser.add_argument("--frame-id", type=int, default=500)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path,
                        default=Path("runs/roma_coarse_airport10_tile01"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    dop_path = args.sequence / "dop_view.jpg"
    if not dop_path.is_file():
        dop_path = args.sequence / "dop.jpg"
    dop_rgb, dop = resize_dop(dop_path)
    uav_rgb = load_frame(args.sequence / "video.mp4", args.frame_id)
    device = torch.device(args.device)

    matcher = RoMaCoarseMatcher(config["roma_dino_checkpoint"],
                                config["matchanything_checkpoint"]).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    uav = torch.from_numpy(uav_rgb.copy()).permute(2, 0, 1).float()[None].to(device) / 255
    dop_input = dop[None].to(device)
    with torch.inference_mode():
        target_features = matcher.encode_target((dop_input - mean) / std)
        result = matcher.match_with_target((uav - mean) / std, target_features,
                                           dop_input.shape[-2:])

    source_h, source_w = result["grid_size"]
    probabilities = result["match_prob"][0].float().cpu().numpy()
    certainty = result["certainty"][0].float().cpu().numpy()
    certainty_grid = certainty.reshape(source_h, source_w)
    target_distribution = np.average(probabilities, axis=0, weights=certainty + 1e-8)
    target_heatmap = target_distribution.reshape(64, 64)
    max_index = probabilities.argmax(axis=1)
    pred_x = (max_index % 64).reshape(source_h, source_w)
    pred_y = (max_index // 64).reshape(source_h, source_w)

    args.output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(17, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.12))
    ax_uav = fig.add_subplot(grid[0, 0])
    ax_dop = fig.add_subplot(grid[0, 1])
    ax_aggregate = fig.add_subplot(grid[1, 0])
    subgrid = grid[1, 1].subgridspec(3, 4, wspace=.12, hspace=.3)

    ax_uav.imshow(uav_rgb)
    ax_uav.imshow(cv2.resize(certainty_grid, (uav_rgb.shape[1], uav_rgb.shape[0]),
                             interpolation=cv2.INTER_CUBIC), cmap="magma", alpha=.48,
                  vmin=0, vmax=1)
    ax_uav.set_title(f"UAV frame {args.frame_id} | RoMa certainty")
    ax_uav.axis("off")

    ax_dop.imshow(dop_rgb)
    overlay = cv2.resize(target_heatmap, (dop_rgb.shape[1], dop_rgb.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
    ax_dop.imshow(overlay, cmap="magma", alpha=.55)
    # Thin the dense patch map for readability; each marker is a UAV patch's MAP anchor.
    step_y, step_x = max(1, source_h // 9), max(1, source_w // 14)
    sy, sx = np.mgrid[0:source_h:step_y, 0:source_w:step_x]
    px = pred_x[::step_y, ::step_x].ravel() * (dop_rgb.shape[1] - 1) / 63
    py = pred_y[::step_y, ::step_x].ravel() * (dop_rgb.shape[0] - 1) / 63
    conf = certainty_grid[::step_y, ::step_x].ravel()
    ax_dop.scatter(px, py, c=conf, cmap="cool", vmin=0, vmax=1,
                   s=18, edgecolors="white", linewidths=.25)
    ax_dop.set_title("DOP | confidence-weighted 64×64 matches")
    ax_dop.axis("off")

    im = ax_aggregate.imshow(target_heatmap, cmap="magma", origin="upper")
    ax_aggregate.set_title("All UAV patches → satellite 64×64 distribution")
    ax_aggregate.set_xlabel("Satellite anchor X")
    ax_aggregate.set_ylabel("Satellite anchor Y")
    fig.colorbar(im, ax=ax_aggregate, fraction=.046, pad=.04)

    ys = np.linspace(0, source_h - 1, 3, dtype=int)
    xs = np.linspace(0, source_w - 1, 4, dtype=int)
    for row, syi in enumerate(ys):
        for col, sxi in enumerate(xs):
            index = syi * source_w + sxi
            heat = probabilities[index].reshape(64, 64)
            ax = fig.add_subplot(subgrid[row, col])
            ax.imshow(heat, cmap="magma", origin="upper", vmin=0,
                      vmax=max(float(heat.max()), 1e-6))
            ax.set_title(f"UAV patch ({sxi},{syi}) c={certainty_grid[syi,sxi]:.2f}",
                         fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"MatchAnything RoMa coarse | airport tile | frame {args.frame_id}",
                 fontsize=16)
    output_path = args.output / f"frame_{args.frame_id:06d}_roma_coarse.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    np.savez_compressed(args.output / f"frame_{args.frame_id:06d}_roma_coarse.npz",
                        match_prob=probabilities, certainty=certainty_grid,
                        aggregate_heatmap=target_heatmap,
                        grid_size=np.array([source_h, source_w]))
    print(f"RoMa grid: {source_h}x{source_w}; mean certainty={certainty.mean():.3f}; "
          f"high-certainty patches={(certainty >= .5).mean():.1%}")
    print(output_path)


if __name__ == "__main__":
    main()

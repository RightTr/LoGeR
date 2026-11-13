"""Evaluate SGGA B/C against LoGeR* on sampled full sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from eval.datasets_preprocess.movingdrone import load_frames
from loger.models.pi3 import Pi3
from loger.models.satellite_encoder import SatelliteEncoder


def rigid_align(source: np.ndarray, target: np.ndarray):
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    translation = target_mean - source_mean @ rotation
    return source @ rotation + translation


def sim3_align(source: np.ndarray, target: np.ndarray):
    source_mean, target_mean = source.mean(0), target.mean(0)
    src, dst = source - source_mean, target - target_mean
    u, singular, vt = np.linalg.svd(src.T @ dst)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = float((singular * np.diag(correction)).sum() /
                  np.square(src).sum().clip(1e-12))
    return scale * source @ rotation + (target_mean - scale * source_mean @ rotation), scale


def metrics(prediction: np.ndarray, target: np.ndarray):
    se3 = rigid_align(prediction, target)
    sim3, scale = sim3_align(prediction, target)
    pred_rpe = np.diff(se3, axis=0)
    gt_rpe = np.diff(target, axis=0)
    return {
        "se3_ate_rmse_m": float(np.sqrt(np.square(se3 - target).sum(-1).mean())),
        "sim3_ate_rmse_m": float(np.sqrt(np.square(sim3 - target).sum(-1).mean())),
        "sim3_scale_required": scale,
        "se3_rpe_translation_rmse_m": float(np.sqrt(np.square(pred_rpe - gt_rpe).sum(-1).mean())),
    }, se3, sim3


def load_region_model(config, checkpoint_path, device):
    model_config = yaml.safe_load(Path(config["model_config"]).read_text())["model"]
    model_config.pop("se3", None)
    model_config.update(sgga_enabled=True,
                        r2la_enabled=bool(config["r2la_enabled"]),
                        decoder_gradient_checkpointing=True,
                        ttt_gradient_checkpointing=False)
    model = Pi3(**model_config)
    base = torch.load(config["loger_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(base.get("model_state_dict", base.get("state_dict", base)), strict=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sgga_state = state.get("sgga", state.get("satellite_region"))
    if sgga_state is None:
        raise KeyError("Checkpoint has neither 'sgga' nor legacy 'satellite_region' weights")
    if "satellite_region" in state and "sgga" not in state:
        legacy = dict(sgga_state)
        sgga_state = {key: value for key, value in legacy.items()
                      if not key.startswith(("local_projection.", "region_down.",
                                             "stages.", "blocks.", "region_up.",
                                             "local_up.", "region_residual_scale",
                                             "local_residual_scale"))}
        if "joint_gates" in sgga_state and "sgga_gates" not in sgga_state:
            sgga_state["sgga_gates"] = sgga_state.pop("joint_gates")
        if model.r2la_enabled:
            legacy_r2la = {}
            for key, value in legacy.items():
                if key.startswith("local_projection.") or key.startswith("region_down."):
                    legacy_r2la[key] = value
                elif key.startswith("stages."):
                    parts = key.split(".", 2)
                    if len(parts) == 3 and int(parts[1]) < 4:
                        legacy_r2la[f"blocks.{parts[1]}.{parts[2]}"] = value
                elif key.startswith("region_up."):
                    parts = key.split(".", 2)
                    if len(parts) == 3 and int(parts[1]) < 4:
                        legacy_r2la[f"region_up.{parts[1]}.{parts[2]}"] = value
                elif key == "region_residual_scale":
                    legacy_r2la[key] = value[:4]
            model.r2la.load_state_dict(legacy_r2la, strict=True)
    model.sgga.load_state_dict(sgga_state, strict=True)
    if model.r2la_enabled and "r2la" in state:
        model.r2la.load_state_dict(state["r2la"], strict=True)
    model.to(device).eval()
    return model


def encode_dop(sequence: Path, device, batch_size=4):
    encoder = SatelliteEncoder(pretrained=True, tile_batch_size=batch_size).to(device).eval()
    features = encoder.encode_path(sequence / "dop_view.jpg", device).cpu().half()
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features


def run(model, images, device, satellite_features=None, use_sgga=False):
    kwargs = {"satellite_features": satellite_features} if use_sgga else {}
    model.sgga_enabled = bool(use_sgga)
    model.eval()
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model(images[None].to(device), window_size=6, overlap_size=2,
                           num_iterations=1, no_detach=True, se3=True, **kwargs)
    centers = prediction["camera_poses"][0, :, :3, 3].float().cpu().numpy()
    del prediction
    return centers.astype(np.float64)


def evaluate_sequence(sequence: Path, model, device, frames: int, output: Path):
    images, target, source_ids = load_frames(sequence, frames, size=(448, 252))
    satellite_features = encode_dop(sequence, device)
    baseline = run(model, images, device, use_sgga=False)
    region = run(model, images, device, satellite_features, use_sgga=True)
    base_metrics, base_se3, base_sim3 = metrics(baseline, target)
    region_metrics, region_se3, region_sim3 = metrics(region, target)
    result = {
        "sequence": str(sequence), "sampled_frames": len(source_ids),
        "source_frame_ids": source_ids.tolist(),
        "LoGeR_star": base_metrics,
        "SGGA": region_metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    label = "_".join(sequence.parts[-2:])
    (output / f"{label}.json").write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, (aligned_base, aligned_region, title) in zip(
            axes, ((base_se3, region_se3, "SE(3) aligned"),
                   (base_sim3, region_sim3, "Sim(3) aligned"))):
        ax.plot(target[:, 0], target[:, 1], "k--", lw=2, label="GT")
        ax.plot(aligned_base[:, 0], aligned_base[:, 1], color="#4c78a8", label="LoGeR*")
        ax.plot(aligned_region[:, 0], aligned_region[:, 1], color="#e45756",
                label="SGGA")
        stride = max(1, len(source_ids) // 12)
        for i in range(0, len(source_ids), stride):
            ax.annotate(str(int(source_ids[i])), target[i, :2], fontsize=7)
        ax.set_title(title); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.axis("equal"); ax.grid(alpha=.25); ax.legend()
    fig.suptitle(f"{label} | {len(source_ids)} sampled frames")
    fig.savefig(output / f"{label}.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"sequence": label, "frames": len(source_ids),
                      "LoGeR_star": base_metrics, "SGGA": region_metrics}, indent=2),
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--frames", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--sequences", nargs="*", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint = args.checkpoint or str(Path(config["output"]) / "best.pt")
    split_path = Path(args.split or config["split_file"])
    split = json.loads(split_path.read_text())
    sequences = [Path(path) for path in (args.sequences or split["val"])]
    device = torch.device(config["device"])
    model = load_region_model(config, checkpoint, device)
    for count in args.frames:
        for sequence in sequences:
            evaluate_sequence(sequence, model, device, int(count),
                              Path(config["output"]) / f"eval_{count}")


if __name__ == "__main__":
    main()

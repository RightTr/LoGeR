"""Run LoGeR on MovingDrone sequences and evaluate camera centers after Sim3 alignment."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.pi3_adapter import load_pi3_model
from eval.datasets_preprocess.movingdrone import find_scenes, group_scenes, load_frames, result_dir
from loger.utils.satellite import SatelliteTileManager
from tools.visualize_sca import dop_image, overlay
from eval.uav_satelitte.plot_results import (
    plot_ablation,
    plot_counts,
)


def sim3(src, dst):
    a, b = src.mean(0), dst.mean(0)
    u, s, vt = np.linalg.svd((src - a).T @ (dst - b) / len(src))
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    r = vt.T @ d @ u.T
    scale = np.sum(s * np.diag(d)) / np.mean(np.sum((src - a) ** 2, axis=1))
    return scale, r, b - scale * r @ a


def run_sca_ablation(args, names):
    from matplotlib import pyplot as plt
    from torch.utils.tensorboard import SummaryWriter

    model, _ = load_pi3_model(str(ROOT / 'ckpts/LoGeR/latest.pt'),
                              str(ROOT / 'ckpts/LoGeR/original_config.yaml'),
                              torch.device(args.gpu), sca_enabled=True)
    weights = torch.load(args.sca_weights, map_location='cpu', weights_only=True)
    model.sca.load_state_dict(weights)
    manifest = json.loads(args.dop_manifest.read_text())
    writer = SummaryWriter(str(args.output / 'tensorboard'))
    summary = {}
    for name in names:
        images, gt, ids = load_frames(args.scenes / name, args.frames)
        results = {}
        for method in ('E0', 'E1', 'E2'):
            model.sca_enabled = method != 'E0'
            features = None
            if method != 'E0':
                with SatelliteTileManager(manifest[name]['correct' if method == 'E1' else 'wrong']) as manager:
                    tiles = manager.get_all_tiles()
                with torch.inference_mode():
                    features = model.encode_satellite(tiles)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(images[None], satellite_features=features, window_size=8,
                             overlap_size=2, num_iterations=1, sim3=True)
            poses = pred['camera_poses'][0].float().cpu().numpy()
            centers = poses[:, :3, 3].astype(np.float64)
            scale, rotation, translation = sim3(centers, gt)
            aligned = scale * (centers @ rotation.T) + translation
            errors = np.linalg.norm(aligned - gt, axis=1)
            relative = np.linalg.norm(np.diff(aligned, axis=0) - np.diff(gt, axis=0), axis=1)
            results[method] = {'ATE': float(np.sqrt(np.mean(errors ** 2))),
                               'RPE': float(np.sqrt(np.mean(relative ** 2))),
                               'aligned': aligned.tolist()}
            writer.add_scalar(f'{name}/ATE/{method}', results[method]['ATE'], args.frames)
            writer.add_scalar(f'{name}/RPE/{method}', results[method]['RPE'], args.frames)
            if method == 'E1':
                dop, size = dop_image(manifest[name]['correct'])
                attention = pred['satellite_attention']['frame'][0, 0].float().cpu().numpy()
                coords = features['pool_map_coords'][0].cpu().numpy()
                writer.add_image(f'{name}/satellite_attention',
                                 overlay(dop, size, attention, coords), args.frames, dataformats='HWC')
        figure, axis = plt.subplots()
        axis.plot(gt[:, 0], gt[:, 1], label='GT')
        for method in ('E0', 'E1', 'E2'):
            aligned = np.asarray(results[method]['aligned'])
            axis.plot(aligned[:, 0], aligned[:, 1], label=method)
        axis.legend()
        axis.set_aspect('equal')
        writer.add_figure(f'{name}/trajectory', figure, args.frames)
        plt.close(figure)
        summary[name] = results
    writer.close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'sca_ablation.json').write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', type=Path, default=ROOT.parent / 'MovingDrone_20/scenes')
    p.add_argument('--names', nargs='+', help='Scene names; default: every scene directory')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    p.add_argument('--frames', type=int, default=16, choices=(16, 64))
    p.add_argument('--gpu', default='cuda:0')
    p.add_argument('--sca-weights', type=Path)
    p.add_argument('--dop-manifest', type=Path, help='JSON: scene name -> correct/wrong GeoTIFF paths')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = args.names or find_scenes(args.scenes)
    if args.sca_weights:
        run_sca_ablation(args, names)
        return
    model, _ = load_pi3_model(str(ROOT / 'ckpts/LoGeR/latest.pt'), str(ROOT / 'ckpts/LoGeR/original_config.yaml'), torch.device(args.gpu))
    summary_file = args.output / 'summary.json'
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    for name in names:
        print(f'Running {name}', flush=True)
        scene_dir = args.scenes / name
        images, gt, ids = load_frames(scene_dir, args.frames)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(images[None], window_size=len(images), overlap_size=2, num_iterations=1)
        poses = pred['camera_poses'][0].float().cpu().numpy()
        centers = poses[:, :3, 3].astype(np.float64)
        scale, rot, trans = sim3(centers, gt)
        aligned = scale * (centers @ rot.T) + trans
        errors = np.linalg.norm(aligned - gt, axis=1)
        relative_errors = np.linalg.norm(np.diff(aligned, axis=0) - np.diff(gt, axis=0), axis=1)
        aligned_poses = poses.astype(np.float64).copy()
        aligned_poses[:, :3, :3] = rot @ aligned_poses[:, :3, :3]
        aligned_poses[:, :3, 3] = aligned
        out = result_dir(args.output, name, args.frames)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'poses.npz', frame_ids=ids, predicted=poses, aligned=aligned_poses, gt_centers=gt, sim3_scale=scale, sim3_rotation=rot, sim3_translation=trans, errors_m=errors)
        with (out / 'frame_errors.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'position_error_m'])
            writer.writerows(zip(ids, errors))
        path_length = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
        ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
        result = {'frames': len(ids), 'frame_ids': ids.tolist(), 'ate_rmse_m': ate_rmse, 'ate_mean_m': float(errors.mean()), 'ate_median_m': float(np.median(errors)), 'ate_p90_m': float(np.percentile(errors, 90)), 'ate_rmse_over_path_pct': 100 * ate_rmse / path_length, 'relative_translation_rmse_m': float(np.sqrt(np.mean(relative_errors ** 2))), 'sim3_scale': float(scale), 'gt_path_length_m': path_length}
        (out / 'metrics.json').write_text(json.dumps(result, indent=2))
        summary[name] = result
        summary_file.write_text(json.dumps(summary, indent=2))
        print(name, result, flush=True)
        del pred
        torch.cuda.empty_cache()

    for sequence in group_scenes(names):
        group = args.output / sequence / str(args.frames)
        for figure in plot_ablation(group):
            print(f'Comparison: {figure}', flush=True)
        seq_root = group.parent
        other = seq_root / ('64' if args.frames == 16 else '16')
        if other.is_dir():
            print('Metrics:', plot_counts(
                seq_root / '16', seq_root / '64', seq_root / 'compare_16_64.png'
            ), flush=True)
            from eval.uav_satelitte.plot_results import plot_trajectories
            print('Trajectories:', plot_trajectories(
                seq_root / '16', seq_root / '64', seq_root / 'trajectory_16_64.png'
            ), flush=True)


if __name__ == '__main__':
    main()

"""Create PNG previews from saved MovingDrone poses and reconstruction PLYs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

VARIANT_LABELS = {
    'baseline': 'Base', 'height_fixed': 'H-fix', 'height_high': 'H-high',
    'height_low': 'H-low', 'path_crossing': 'Cross', 'path_straight': 'Straight',
    'reference': 'Ref',
}


def read_ply(path):
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
    with path.open('rb') as f:
        while f.readline().strip() != b'end_header':
            pass
        data = np.fromfile(f, dtype=dtype)
    xyz = np.column_stack([data[k] for k in ('x', 'y', 'z')])
    rgb = np.column_stack([data[k] for k in ('r', 'g', 'b')]) / 255
    return xyz, rgb


def equal_axes(ax, xyz):
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    mid = (lo + hi) / 2
    radius = max((hi - lo).max() / 2, 1)
    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_scene(directory, scenes_root=None):
    data = np.load(directory / 'poses.npz')
    gt, pred = data['gt_centers'], data['aligned'][:, :3, 3]
    if scenes_root is None:
        scenes_root = Path(__file__).resolve().parents[3] / 'MovingDrone_20' / 'scenes'
    csv_path = Path(scenes_root) / directory.name / 'poses.csv'
    with csv_path.open() as f:
        full_gt = np.array([[float(row[k]) for k in ('x', 'y', 'z')] for row in csv.DictReader(f)])
    distance = np.linalg.norm(np.diff(full_gt, axis=0), axis=1).sum()
    height_min, height_max = full_gt[:, 2].min(), full_gt[:, 2].max()
    info = f'Full GT distance {distance:.1f} m | height {height_min:.1f}–{height_max:.1f} m'
    origin = gt.mean(0)
    gt, pred = gt - origin, pred - origin
    errors = data['errors_m']
    xy_rmse = np.sqrt(np.mean(np.sum((pred[:, :2] - gt[:, :2]) ** 2, axis=1)))
    xyz_rmse = np.sqrt(np.mean(errors ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, ix, iy, xlabel, ylabel in [(axes[0], 0, 1, 'X (m)', 'Y (m)'), (axes[1], 0, 2, 'X (m)', 'Z (m)')]:
        ax.plot(gt[:, ix], gt[:, iy], '-o', color='black', markersize=3, label='GT')
        ax.plot(pred[:, ix], pred[:, iy], '-o', color='#e45756', markersize=3, label='LoGeR + Sim3')
        for a, b in zip(gt, pred):
            ax.plot([a[ix], b[ix]], [a[iy], b[iy]], color='#e45756', alpha=.25, linewidth=.8)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.axis('equal')
        ax.grid(alpha=.25)
    axes[0].legend()
    axes[0].set_title(f'XY ATE RMSE: {xy_rmse:.2f} m')
    axes[1].set_title(f'XYZ ATE RMSE: {xyz_rmse:.2f} m')
    fig.suptitle(f'{directory.name} | trajectory comparison\n{info}')
    fig.savefig(directory / 'trajectory_comparison.png', dpi=180)
    plt.close(fig)

    xyz, rgb = read_ply(directory / 'reconstruction_aligned.ply')
    xyz -= origin
    valid = np.isfinite(xyz).all(1)
    xyz, rgb = xyz[valid][::8], rgb[valid][::8]
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(121)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb, s=.2, rasterized=True)
    ax.plot(gt[:, 0], gt[:, 1], '-o', color='black', markersize=2, label='GT')
    ax.plot(pred[:, 0], pred[:, 1], '-o', color='#e45756', markersize=2, label='LoGeR + Sim3')
    ax.set(xlabel='X (m)', ylabel='Y (m)', title='Top view')
    ax.axis('equal')
    ax.legend()
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=.15, alpha=.7, rasterized=True)
    ax.plot(*gt.T, color='black', linewidth=2)
    ax.plot(*pred.T, color='#e45756', linewidth=2)
    equal_axes(ax, np.concatenate([xyz, gt, pred]))
    ax.view_init(elev=28, azim=-65)
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='3D view')
    fig.suptitle(f'{directory.name} | aligned reconstruction (sampled points)\n{info}')
    fig.savefig(directory / 'reconstruction_preview.png', dpi=180)
    plt.close(fig)


def plot_ablation(results_dir):
    """Compare evaluation metrics across the variant folders in one sequence."""
    import json

    results_dir = Path(results_dir)
    entries = []
    for metrics_path in sorted(results_dir.glob('*/metrics.json')):
        with metrics_path.open() as f:
            metrics = json.load(f)
        entries.append((metrics_path.parent.name, metrics))
    if len(entries) < 2:
        raise ValueError(f'Need at least two evaluated variants under {results_dir}')

    labels = [VARIANT_LABELS.get(name, name.replace('_', ' ')) for name, _ in entries]
    ate = [m['ate_rmse_m'] for _, m in entries]
    relative = [m['relative_translation_rmse_m'] for _, m in entries]
    normalized = [m['ate_rmse_over_path_pct'] for _, m in entries]
    x = np.arange(len(entries))
    width = 0.26

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    series = [
        (ate, 'ATE RMSE (m)', '#4c78a8'),
        (relative, 'Relative translation RMSE (m)', '#f58518'),
        (normalized, 'ATE RMSE / path length (%)', '#54a24b'),
    ]
    for ax, (values, title, color) in zip(axes, series):
        bars = ax.bar(x, values, width=0.72, color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=35, ha='right')
        ax.grid(axis='y', alpha=.25)
        ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
    fig.suptitle(f'{results_dir.name} | ablation comparison')
    output = results_dir / 'metrics.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)

    variants = [path.parent for path in sorted(results_dir.glob('*/reconstruction_aligned.ply'))]
    if variants:
        cols = min(3, len(variants))
        rows = int(np.ceil(len(variants) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False, constrained_layout=True)
        for ax, directory in zip(axes.flat, variants):
            data = np.load(directory / 'poses.npz')
            gt = data['gt_centers']
            pred = data['aligned'][:, :3, 3]
            origin = gt.mean(0)
            xyz, rgb = read_ply(directory / 'reconstruction_aligned.ply')
            xyz -= origin
            valid = np.isfinite(xyz).all(1)
            xyz, rgb = xyz[valid][::12], rgb[valid][::12]
            ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb, s=.2, rasterized=True, alpha=.7)
            ax.plot(gt[:, 0] - origin[0], gt[:, 1] - origin[1], '-o', color='black', markersize=2, label='GT')
            ax.plot(pred[:, 0] - origin[0], pred[:, 1] - origin[1], '-o', color='#e45756', markersize=2, label='LoGeR')
            ax.set_title(VARIANT_LABELS.get(directory.name, directory.name.replace('_', ' ')))
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.axis('equal')
            ax.grid(alpha=.2)
        for ax in axes.flat[len(variants):]:
            ax.remove()
        axes.flat[0].legend()
        fig.suptitle(f'{results_dir.name} | reconstruction comparison')
        vis_output = results_dir / 'recon.png'
        fig.savefig(vis_output, dpi=180)
        plt.close(fig)
        return output, vis_output
    return (output,)


def plot_counts(results16, results64, output):
    """Compare both sampling densities for matching ablation variants."""
    import json

    results16, results64, output = Path(results16), Path(results64), Path(output)
    names = sorted(p.parent.name for p in results16.glob('*/metrics.json')
                   if (results64 / p.parent.name / 'metrics.json').is_file())
    if not names:
        raise ValueError('No variants have both 16-frame and 64-frame results')
    metrics = {}
    for n in names:
        metrics[n] = []
        for root in (results16, results64):
            with (root / n / 'metrics.json').open() as f:
                metrics[n].append(json.load(f))
    labels = [VARIANT_LABELS.get(n, n.replace('_', ' ')) for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, key, title in zip(axes, ('ate_rmse_m','relative_translation_rmse_m'),
                              ('ATE RMSE (m)','Relative translation RMSE (m)')):
        w=.36
        b16=ax.bar(x-w/2,[metrics[n][0][key] for n in names],w,label='16 frames',color='#4c78a8')
        b64=ax.bar(x+w/2,[metrics[n][1][key] for n in names],w,label='64 frames',color='#f58518')
        ax.bar_label(b16,fmt='%.1f',fontsize=7,padding=2)
        ax.bar_label(b64,fmt='%.1f',fontsize=7,padding=2)
        ax.set_title(title); ax.set_xticks(x,labels,rotation=35,ha='right'); ax.grid(axis='y',alpha=.25); ax.legend()
    fig.suptitle(f'{output.parent.name} | 16 vs 64 frame comparison')
    fig.savefig(output,dpi=180); plt.close(fig)
    return output


def plot_trajectories(results16, results64, output):
    """Compare GT and 16/64-frame estimated trajectories for each variant."""
    results16, results64, output = Path(results16), Path(results64), Path(output)
    names = sorted(p.parent.name for p in results16.glob('*/reconstruction_aligned.ply')
                   if (results64 / p.parent.name / 'reconstruction_aligned.ply').is_file())
    if not names:
        raise ValueError('No variants have both 16-frame and 64-frame reconstructions')
    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False,
                             constrained_layout=True)
    for index, name in enumerate(names):
        ax = axes.flat[index]
        d16, d64 = results16 / name, results64 / name
        data16 = np.load(d16 / 'poses.npz')
        data64 = np.load(d64 / 'poses.npz')
        gt = data64['gt_centers']
        origin = gt.mean(0)
        ax.plot(gt[:,0]-origin[0], gt[:,1]-origin[1], '-o', color='#39ff14', ms=3.2,
                lw=3.2, markeredgecolor='#145214', markeredgewidth=.5, label='GT')
        for data, color, label in ((data16,'#f28e2b','LoGeR 16'),(data64,'#d01c8b','LoGeR 64')):
            estimate = data['aligned'][:, :3, 3]
            ax.plot(estimate[:,0]-origin[0], estimate[:,1]-origin[1], '-o', color=color, ms=2.3, lw=1.5, label=label)
        ax.set_title(VARIANT_LABELS.get(name, name.replace('_', ' ')))
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.axis('equal'); ax.grid(alpha=.2)
        if index == 0: ax.legend(fontsize=8)
    for ax in axes.flat[len(names):]: ax.remove()
    fig.suptitle(f'{output.parent.name} | GT vs LoGeR trajectories (16/64 frames)')
    fig.savefig(output, dpi=180); plt.close(fig)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path, nargs='?', default=Path(__file__).resolve().parent / 'results')
    p.add_argument('--scenes', type=Path, default=None, help='Root directory containing scene folders and poses.csv')
    p.add_argument('--ablation', action='store_true', help='Also plot a metric comparison across variant folders')
    args = p.parse_args()
    for path in sorted(args.results.glob('*/poses.npz')):
        plot_scene(path.parent, args.scenes)
        print(path.parent)
    if args.ablation:
        for output in plot_ablation(args.results):
            print(output)


if __name__ == '__main__':
    main()

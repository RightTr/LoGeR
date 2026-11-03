"""LoGeR geometry losses."""

import torch
import torch.nn.functional as F


def depth_to_pointmap(depth, intrinsics):
    batch, frames, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    rays = torch.linalg.solve(intrinsics, pixels.reshape(-1, 3).T.expand(batch, frames, 3, -1))
    points = rays.transpose(-1, -2).reshape(batch, frames, height, width, 3) * depth[..., None]
    valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(points).all(-1)
    return points, valid


def rotation_geodesic(predicted, target):
    relative = predicted.transpose(-1, -2) @ target
    skew = torch.stack(
        (relative[..., 2, 1] - relative[..., 1, 2],
         relative[..., 0, 2] - relative[..., 2, 0],
         relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
    sine = torch.linalg.vector_norm(skew, dim=-1) * 0.5
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    return torch.atan2(sine, cosine)


def estimate_scale(predicted, target, valid):
    """Detached positive weighted-L1 scale, one value per input sequence."""
    depth = target[..., 2]
    count = valid.sum((1, 2, 3)).clamp_min(1)
    mean_depth = (depth * valid).sum((1, 2, 3)) / count
    point_weight = valid / torch.maximum(depth, 0.1 * mean_depth[:, None, None, None]).clamp_min(1e-6)
    x, y = predicted.flatten(1), target.flatten(1)
    weights = point_weight[..., None].expand_as(predicted).flatten(1)
    usable = (weights > 0) & (x.abs() > 1e-8) & torch.isfinite(x) & torch.isfinite(y)
    ratios = (y / torch.where(x.abs() > 1e-8, x, torch.ones_like(x))).clamp_min(0)
    ratios = ratios.masked_fill(~usable, float("inf"))
    order = ratios.argsort(dim=1)
    ratios = ratios.gather(1, order)
    masses = torch.where(usable, weights * x.abs(), 0).gather(1, order)
    index = (masses.cumsum(1) >= masses.sum(1, keepdim=True) * 0.5).to(torch.int64).argmax(1, keepdim=True)
    scale = ratios.gather(1, index).squeeze(1)
    scale = torch.where(torch.isfinite(scale), scale.clamp_min(1e-6), torch.ones_like(scale))
    return scale.detach()


def _depth_weights(target, valid):
    depth = target[..., 2]
    mean_depth = (depth * valid).sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1)
    return valid / torch.maximum(depth, 0.1 * mean_depth[:, None, None, None]).clamp_min(1e-6)


def local_pointmap_loss(predicted, target, valid, scale):
    error = (predicted * scale[:, None, None, None, None] - target).abs().mean(-1)
    weights = _depth_weights(target, valid)
    per_sample = (error * weights).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
    return per_sample.mean()


def relative_pose_loss(predicted, target, scale, valid_frames, lambda_rotation, lambda_translation):
    frames = valid_frames.shape[1]
    indices = torch.triu_indices(frames, frames, offset=1, device=predicted.device)
    first, second = indices
    pair_valid = valid_frames[:, first] & valid_frames[:, second]
    pred_rel = torch.linalg.inv(predicted[:, first]) @ predicted[:, second]
    target_rel = torch.linalg.inv(target[:, first]) @ target[:, second]
    rotation = rotation_geodesic(pred_rel[..., :3, :3], target_rel[..., :3, :3])
    translation = F.huber_loss(
        pred_rel[..., :3, 3] * scale[:, None, None], target_rel[..., :3, 3],
        reduction="none", delta=0.1).mean(-1)
    denominator = pair_valid.sum(1).clamp_min(1)
    rotation = (rotation * pair_valid).sum(1) / denominator
    translation = (translation * pair_valid).sum(1) / denominator
    sample_valid = pair_valid.any(1)
    if sample_valid.any():
        rotation = rotation[sample_valid].mean()
        translation = translation[sample_valid].mean()
    else:
        rotation = predicted.sum() * 0
        translation = predicted.sum() * 0
    return lambda_rotation * rotation + lambda_translation * translation, rotation, translation


def global_pointmap_loss(predicted, predicted_poses, target, target_poses, valid, scale):
    pred_rel = torch.linalg.inv(predicted_poses[:, :1]) @ predicted_poses
    gt_rel = torch.linalg.inv(target_poses[:, :1]) @ target_poses
    pred_rel = pred_rel.clone()
    pred_rel[..., :3, 3] *= scale[:, None, None]
    pred_world = torch.einsum("bnij,bnhwj->bnhwi", pred_rel[..., :3, :3], predicted)
    pred_world = pred_world * scale[:, None, None, None, None] + pred_rel[:, :, None, None, :3, 3]
    gt_world = torch.einsum("bnij,bnhwj->bnhwi", gt_rel[..., :3, :3], target)
    gt_world = gt_world + gt_rel[:, :, None, None, :3, 3]
    error = (pred_world - gt_world).abs().mean(-1)
    weights = _depth_weights(target, valid)
    per_sample = (error * weights).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
    return per_sample.mean()


class LoGeRLoss(torch.nn.Module):
    def __init__(self, lambda_global=1.0, lambda_rotation=0.1, lambda_translation=10.0):
        super().__init__()
        self.lambda_global = lambda_global
        self.lambda_rotation = lambda_rotation
        self.lambda_translation = lambda_translation

    def forward(self, predictions, targets):
        predicted = predictions["local_points"]
        target, valid = depth_to_pointmap(targets["depth"], targets["intrinsics"])

        poses, gt_poses = predictions["camera_poses"], targets["poses"]
        valid_frames = torch.isfinite(poses).all((-1, -2)) & torch.isfinite(gt_poses).all((-1, -2))
        valid &= valid_frames[..., None, None] & torch.isfinite(predicted).all(-1)
        sample_valid = valid.flatten(1).any(1)
        predicted, target, valid = predicted[sample_valid], target[sample_valid], valid[sample_valid]
        poses, gt_poses = poses[sample_valid], gt_poses[sample_valid]
        valid_frames = valid_frames[sample_valid]
        identity = torch.eye(4, device=poses.device, dtype=poses.dtype)
        poses = torch.where(valid_frames[..., None, None], poses, identity)
        gt_poses = torch.where(valid_frames[..., None, None], gt_poses, identity)
        target = torch.where(valid[..., None], target, 0)
        predicted = torch.where(valid[..., None], predicted, 0)
        scale = estimate_scale(predicted, target, valid)
        local = local_pointmap_loss(predicted, target, valid, scale)
        pose, rotation, translation = relative_pose_loss(
            poses, gt_poses, scale, valid_frames, self.lambda_rotation, self.lambda_translation)
        global_loss = global_pointmap_loss(predicted, poses, target, gt_poses, valid, scale)
        total = local + pose + self.lambda_global * global_loss
        return {"total": total, "local": local, "pose": pose, "rotation": rotation,
                "translation": translation, "global": global_loss, "sequence_scale": scale}

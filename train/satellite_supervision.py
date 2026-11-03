"""Project sampled UAV depth into DOP token occupancy."""

import torch
import torch.nn.functional as F


def project(depth, intrinsics, poses, world_to_dop, sample_size=8):
    batch, frames, height, width = depth.shape
    y = (torch.arange(sample_size, device=depth.device) + 0.5) * height / sample_size
    x = (torch.arange(sample_size, device=depth.device) + 0.5) * width / sample_size
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pixels = torch.stack((xx.flatten(), yy.flatten(), torch.ones_like(xx).flatten()), dim=-1)
    indices = pixels[:, :2].long()
    distance = depth[..., indices[:, 1], indices[:, 0]]
    rays = torch.linalg.solve(intrinsics, pixels.T.expand(batch, frames, 3, -1)).transpose(-1, -2)
    camera = rays * distance[..., None]
    world = camera @ poses[..., :3, :3].transpose(-1, -2) + poses[..., None, :3, 3]
    xy1 = torch.cat((world[..., :2], torch.ones_like(world[..., :1])), dim=-1)
    dop = xy1 @ world_to_dop[..., None, :3, :3].transpose(-1, -2)
    dop = dop[..., :2] / dop[..., 2:].clamp_min(1e-8)
    valid = distance.isfinite() & (distance > 0) & dop.isfinite().all(dim=-1)
    return dop, valid


def occupancy(points, valid, origins, token_mask, width, height):
    batch, frames, count, _ = points.shape
    origins = origins.to(points.device)
    inside = points[..., None, :] - origins[None, None, None]
    x, y = inside.unbind(-1)
    covered = valid[..., None] & (x >= 0) & (x < 518) & (y >= 0) & (y < 518)
    covered &= (points[..., 0, None] >= 0) & (points[..., 0, None] < width)
    covered &= (points[..., 1, None] >= 0) & (points[..., 1, None] < height)
    col = (x * 16 / 518).long().clamp(0, 15)
    row = (y * 16 / 518).long().clamp(0, 15)
    tile = torch.arange(len(origins), device=points.device)
    index = tile[None, None, None] * 256 + row * 16 + col
    covered &= token_mask.to(points.device).flatten()[index]
    counts = torch.zeros(batch, frames, len(origins) * 256, device=points.device)
    counts.scatter_add_(-1, index.reshape(batch, frames, -1), covered.float().reshape(batch, frames, -1))
    frame_valid = counts.sum(-1) > 0
    frame = counts / counts.sum(-1, keepdim=True).clamp_min(1)
    chunk = counts.sum(dim=1)
    chunk = chunk / chunk.sum(-1, keepdim=True).clamp_min(1)
    return frame, chunk, frame_valid


def kl(target, prediction, valid):
    loss = F.kl_div(prediction.clamp_min(1e-12).log(), target, reduction="none").sum(-1)
    return loss[valid].mean()

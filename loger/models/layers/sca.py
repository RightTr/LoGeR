"""Single-query Mid-to-Fine satellite cross-attention."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCA(nn.Module):
    def __init__(
        self,
        uav_dim,
        satellite_dim,
        attn_dim=256,
        num_heads=8,
        gate_init=0.01,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.query_norm = nn.LayerNorm(uav_dim)
        self.satellite_norm = nn.LayerNorm(satellite_dim)
        self.query = nn.Linear(uav_dim, attn_dim, bias=False)
        self.key = nn.Linear(satellite_dim, attn_dim, bias=False)
        self.value = nn.Linear(satellite_dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, attn_dim)
        self.level_embedding = nn.Parameter(torch.zeros(2, satellite_dim))
        self.lambda_mid = nn.Parameter(torch.tensor(0.4))
        self.register_buffer("frequencies", torch.logspace(0, 2, satellite_dim // 4), persistent=False)
        self.film_norm = nn.LayerNorm(uav_dim)
        self.film = nn.Sequential(nn.Linear(attn_dim, uav_dim), nn.GELU(), nn.Linear(uav_dim, 2 * uav_dim))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def prepare_satellite(self, tokens, coords, valid_mask):
        batch, token_count, channels = tokens.shape
        tile_count = token_count // 256
        source = tokens.reshape(batch * tile_count, 16, 16, channels).permute(0, 3, 1, 2)
        coords = coords.reshape(batch * tile_count, 16, 16, 2).permute(0, 3, 1, 2)
        mask = valid_mask.reshape(batch * tile_count, 1, 16, 16).to(tokens.dtype)
        map_coords = coords.permute(0, 2, 3, 1).reshape(batch, -1, 2).float()
        map_valid = valid_mask.reshape(batch, -1).bool()
        xy_min = map_coords.masked_fill(~map_valid[..., None], torch.inf).amin(dim=1, keepdim=True)
        xy_max = map_coords.masked_fill(~map_valid[..., None], -torch.inf).amax(dim=1, keepdim=True)

        levels = {}
        for level_index, (name, size) in enumerate((("mid", 4), ("fine", 16))):
            pooled_mask = F.adaptive_avg_pool2d(mask, size)
            pooled = F.adaptive_avg_pool2d(source * mask, size) / pooled_mask.clamp_min(1e-6)
            pooled_coords = F.adaptive_avg_pool2d(coords * mask, size) / pooled_mask.clamp_min(1e-6)
            pooled = pooled.flatten(2).transpose(1, 2).reshape(batch, -1, channels)
            pooled_coords = pooled_coords.flatten(2).transpose(1, 2).reshape(batch, -1, 2)
            pooled_mask = pooled_mask.flatten(1).reshape(batch, -1) > 0

            xy = 2 * (pooled_coords.float() - xy_min) / (xy_max - xy_min).clamp_min(1) - 1
            xy = torch.nan_to_num(xy).masked_fill(~pooled_mask[..., None], 0)
            angle = xy[..., None] * self.frequencies.to(xy.device) * math.pi
            position = torch.cat((angle[..., 0, :].sin(), angle[..., 0, :].cos(),
                                  angle[..., 1, :].sin(), angle[..., 1, :].cos()), dim=-1)
            enriched = self.satellite_norm(pooled + position.to(pooled.dtype) + self.level_embedding[level_index])
            key = self.key(enriched).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
            levels[name] = {
                "keys": key,
                "mask": pooled_mask,
                "coords": pooled_coords,
            }
            if name == "fine":
                levels[name]["values"] = self.value(enriched).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        tile = torch.arange(tile_count, device=tokens.device).repeat_interleave(256)
        row = torch.arange(16, device=tokens.device).repeat_interleave(16).repeat(tile_count)
        col = torch.arange(16, device=tokens.device).repeat(tile_count * 16)
        fine_parent = tile * 16 + (row // 4) * 4 + col // 4
        return {"levels": levels, "fine_to_mid": fine_parent}

    @staticmethod
    def _masked_softmax(logits, mask):
        probabilities = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min).softmax(dim=-1)
        return probabilities * mask.any(dim=-1)[:, None, None, None]

    def _scores(self, query, memory, level):
        key = memory["levels"][level]["keys"].float()
        return torch.einsum("bnhd,bhmd->bnhm", query.float(), key) / math.sqrt(self.head_dim)

    def forward(self, hidden, batch_size, num_frames, patch_start_idx, satellite_memory, return_attention=False):
        patches = hidden[:, patch_start_idx:]
        frame_patches = patches.reshape(batch_size, num_frames, -1, hidden.shape[-1])
        query = self.query(self.query_norm(frame_patches.mean(dim=2)))
        query = query.reshape(batch_size, num_frames, self.num_heads, self.head_dim)
        mid = satellite_memory["levels"]["mid"]
        mid_attention = self._masked_softmax(self._scores(query, satellite_memory, "mid"), mid["mask"])
        fine = satellite_memory["levels"]["fine"]
        fine_prior = mid_attention.index_select(-1, satellite_memory["fine_to_mid"])
        fine_logits = self._scores(query, satellite_memory, "fine") + self.lambda_mid * fine_prior.clamp_min(1e-6).log()
        fine_attention = self._masked_softmax(fine_logits, fine["mask"])
        context = torch.einsum("bnhm,bhmd->bnhd", fine_attention, fine["values"].float())
        context = context.reshape(batch_size, num_frames, -1).to(hidden.dtype)
        context = self.output(context)
        gamma, beta = self.film(context).chunk(2, dim=-1)
        patch_delta = gamma.reshape(batch_size * num_frames, 1, -1) * self.film_norm(patches) + beta.reshape(batch_size * num_frames, 1, -1)
        enhanced = hidden.clone()
        enhanced[:, patch_start_idx:] = patches + self.gate * patch_delta

        if return_attention:
            return enhanced, {
                "mid_attention": mid_attention.mean(dim=2),
                "fine_attention": fine_attention.mean(dim=2),
                "mid_coords": mid["coords"],
                "fine_coords": fine["coords"],
            }
        return enhanced, None

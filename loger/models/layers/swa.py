"""Sliding-window attention adapter."""

import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

from .block import BlockRope
from ..dinov2.layers import Mlp
from .attention import FlashAttentionRope


class SWA(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, insert_after, rope):
        super().__init__()
        self.insert_after = list(insert_after)
        self.layers = nn.ModuleList([
            BlockRope(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=rope,
            )
            for _ in self.insert_after
        ])
        self.gate_projs = nn.ModuleList([nn.Linear(dim, 1) for _ in self.insert_after])
        for gate in self.gate_projs:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    @staticmethod
    def remap_state_dict(state_dict):
        prefixes = {
            "swa_layers.": "swa.layers.",
            "swa_gate_projs.": "swa.gate_projs.",
        }
        remapped = OrderedDict()
        for key, value in state_dict.items():
            new_key = next((target + key[len(source):] for source, target in prefixes.items()
                            if key.startswith(source)), key)
            if new_key in remapped:
                raise RuntimeError(f"Checkpoint contains conflicting parameter keys for {new_key!r}")
            remapped[new_key] = value
        if hasattr(state_dict, "_metadata"):
            metadata = OrderedDict()
            for key, value in state_dict._metadata.items():
                new_key = next((target[:-1] + key[len(source) - 1:]
                                for source, target in prefixes.items()
                                if key == source[:-1] or key.startswith(source)), key)
                if new_key in metadata:
                    raise RuntimeError(f"Checkpoint contains conflicting module metadata for {new_key!r}")
                metadata[new_key] = value
            remapped._metadata = metadata
        return remapped

    def forward_layer(self, index, x, xpos, history=None, detach_history=False, turn_off=False):
        """Apply one SWA layer to [B, N, tokens, dim] and update its cache."""
        batch, frames, tokens_per_frame, dim = x.shape
        x_flat = x.reshape(batch, frames * tokens_per_frame, dim)
        use_cache = isinstance(history, dict) and "k" in history and "v" in history

        if use_cache:
            output = self.layers[index].forward_with_kv_cache(
                x_flat, history["k"], history["v"], xpos=xpos
            ).reshape(batch, frames, tokens_per_frame, dim)
        else:
            history_raw = history if history is not None and not isinstance(history, dict) else None
            x_all = torch.cat((history_raw, x), dim=1) if history_raw is not None else x
            total_frames = x_all.shape[1]
            x_all_flat = x_all.reshape(batch, total_frames * tokens_per_frame, dim)
            pos_all = xpos
            if xpos is not None:
                pos_all = xpos.reshape(batch, frames, tokens_per_frame, -1)
                if history_raw is not None:
                    pos_hist = pos_all[:, :1].expand(-1, history_raw.shape[1], -1, -1)
                    pos_all = torch.cat((pos_hist, pos_all), dim=1)
                pos_all = pos_all.reshape(batch, total_frames * tokens_per_frame, -1)
            output = self.layers[index](x_all_flat, xpos=pos_all)
            output = output.reshape(batch, total_frames, tokens_per_frame, dim)
            if history_raw is not None:
                output = output[:, history_raw.shape[1]:]

        gate = torch.nn.functional.silu(self.gate_projs[index](output))
        if turn_off:
            gate = torch.zeros_like(gate)
        updated = x + output * gate

        x_cache = x_flat
        pos_cache = None
        if xpos is not None:
            pos_cache = xpos.reshape(batch, frames, tokens_per_frame, -1)
            pos_cache = pos_cache[:, :1].expand(-1, frames, -1, -1)
            pos_cache = pos_cache.reshape(batch, frames * tokens_per_frame, -1)
        k_new, v_new = self.layers[index].compute_kv_cache(x_cache, xpos=pos_cache)
        if detach_history:
            k_new, v_new = k_new.detach(), v_new.detach()
        return updated, gate, {"k": k_new, "v": v_new}

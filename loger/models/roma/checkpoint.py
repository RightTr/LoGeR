"""Strict loading of the two independent upstream RoMa checkpoints."""

from pathlib import Path
import torch


def _read(path):
    path = Path(path).expanduser()
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _strict_load(module, state, label):
    expected = module.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    mismatched = [key for key in expected.keys() & state.keys()
                  if expected[key].shape != state[key].shape]
    if missing or unexpected or mismatched:
        raise RuntimeError(f"Invalid {label}: missing={missing}, unexpected={unexpected}, "
                           f"shape mismatches={mismatched}")
    module.load_state_dict(state, strict=True)


def load_coarse_weights(model, roma_path):
    weights = _read(roma_path)
    for prefix, module in (("decoder.proj.16.", model.proj["16"]),
                           ("decoder.gps.16.", model.gps["16"])):
        selected = {key[len(prefix):]: value for key, value in weights.items()
                    if key.startswith(prefix)}
        _strict_load(module, selected, prefix)

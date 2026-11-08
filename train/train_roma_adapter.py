"""Two-stage RoMa calibration / geometry training with a caller-supplied loader.

The YAML `data_factory` is `module:function`. The function receives
`(config, phase)` and returns an iterable of batch dictionaries. Each batch has
`images` [B,N,3,H,W] in [0,1], `tiles` [B,T,3,Hs,Ws] ImageNet-normalized,
and `tile_indices` [B,N]. Stage A additionally requires `teacher_gp`
[B,N,512,H/14,W/14], aligned to the UAV crop and produced by official RoMa.
Stage B requires `depth`, `intrinsics`, and `poses`, as accepted by LoGeRLoss.
The factory must return a fresh iterable on each call.
"""

import argparse
import importlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loger.models.pi3 import Pi3
from train.loss import LoGeRLoss


def build_model(config, device):
    with open(config["model_config"]) as stream:
        options = yaml.safe_load(stream)["model"]
    options.update(roma_enabled=True,
                   roma_checkpoint=config["roma_checkpoint"],
                   roma_encode_batch_size=config.get("encode_batch_size", 8))
    model = Pi3(**{k: v for k, v in options.items() if k in Pi3.__init__.__code__.co_varnames})
    checkpoint = torch.load(config["loger_checkpoint"], map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    return model.to(device).eval()


def student_features(model, images, tiles, indices):
    batch, frames, channels, height, width = images.shape
    flat = images.reshape(batch * frames, channels, height, width)
    chunks = []
    with torch.no_grad():
        for images_part in flat.split(model.roma_encode_batch_size):
            images_part = (images_part - model.image_mean) / model.image_std
            chunks.append(model.uav_encoder(images_part, is_training=True)["x_norm_patchtokens"])
        memory = model.encode_roma_satellite(tiles)
        features = memory["patch_tokens"]
        if indices.shape != (batch, frames) or (indices < 0).any() or (indices >= features.shape[1]).any():
            raise ValueError("tile_indices must identify one available tile per UAV frame")
        selected = features[torch.arange(batch, device=indices.device)[:, None], indices]
    return model.roma_encoder.forward_from_features(
        torch.cat(chunks).detach(), selected.reshape(batch * frames, selected.shape[-2], -1).detach(),
        (height // 14, width // 14), memory["grid_size"])["gp_posterior"]


def train(config, stage):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    module_name, function_name = config["data_factory"].split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    model = build_model(config, device)
    model.requires_grad_(False)
    if stage == "calibrate":
        model.roma_encoder.proj.requires_grad_(True)
        model.roma_encoder.satellite_proj.requires_grad_(True)
        model.roma_encoder.train()
        parameters = list(model.roma_encoder.proj.parameters()) + list(model.roma_encoder.satellite_proj.parameters())
    else:
        state = torch.load(config["calibration_checkpoint"], map_location="cpu", weights_only=True)
        model.roma_encoder.load_state_dict(state["roma_encoder"], strict=True)
        model.roma_encoder.eval()
        model.roma_adapter.requires_grad_(True)
        if config.get("adapter_checkpoint"):
            state = torch.load(config["adapter_checkpoint"], map_location="cpu", weights_only=True)
            model.roma_adapter.load_state_dict(state["roma_adapter"], strict=True)
        parameters = list(model.roma_adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.get("learning_rate", 1e-4))
    criterion = LoGeRLoss(lambda_global=config.get("lambda_global", 1.0),
                          lambda_rotation=config.get("lambda_rotation", 0.1),
                          lambda_translation=config.get("lambda_translation", 10.0))
    output_path = Path(config["output_checkpoint"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.get("epochs", 1)):
        for batch in factory(config, "train"):
            images = batch["images"].to(device)
            tiles = batch["tiles"].to(device)
            indices = batch["tile_indices"].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            if stage == "calibrate":
                prediction = student_features(model, images, tiles, indices)
                teacher = batch["teacher_gp"].to(device)
                expected = (images.shape[0], images.shape[1], 512,
                            images.shape[-2] // 14, images.shape[-1] // 14)
                if tuple(teacher.shape) != expected:
                    raise ValueError(f"teacher_gp must match the UAV patch grid {expected}")
                teacher = teacher.flatten(0, 1)
                loss = F.mse_loss(F.normalize(prediction, dim=1), F.normalize(teacher, dim=1))
            else:
                memory = model.encode_roma_satellite(tiles)
                output = model(images, roma_satellite_features=memory, roma_tile_indices=indices,
                               window_size=config.get("window_size", -1),
                               overlap_size=config.get("overlap_size", 1), no_detach=True)
                target = {key: batch[key].to(device) for key in ("depth", "intrinsics", "poses")}
                loss = criterion(output, target)["total"]
            loss.backward()
            optimizer.step()
        torch.save({"roma_encoder": model.roma_encoder.state_dict(),
                    "roma_adapter": model.roma_adapter.state_dict(), "epoch": epoch}, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("calibrate", "geometry"), required=True)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    train(config, args.stage)


if __name__ == "__main__":
    main()

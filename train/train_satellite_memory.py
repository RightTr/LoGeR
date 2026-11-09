"""Train RoMa-guided bidirectional fusion on consecutive LoGeR windows.

The YAML data_factory (module:function) returns batches containing images
[B,N,3,H,W] in [0,1], one RGB tile [B,3,518,518] in [0,1],
tile_id [B], tile_bbox [B,4] in normalized DOP coordinates,
and depth, intrinsics, poses.
N must span at least two windows. The MatchAnything matcher and all existing
LoGeR weights stay frozen; gradients follow the satellite memory across windows.
"""

import argparse
import importlib
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loger.models.pi3 import Pi3
from train.loss import LoGeRLoss


def build_model(config, device):
    with open(config["model_config"]) as stream:
        options = yaml.safe_load(stream)["model"]
    options.update(roma_enabled=True,
                   roma_checkpoint=config["matchanything_checkpoint"],
                   roma_dino_checkpoint=config["roma_dino_checkpoint"])
    model = Pi3(**{key: value for key, value in options.items()
                   if key in Pi3.__init__.__code__.co_varnames})
    checkpoint = torch.load(config["loger_checkpoint"], map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    model.requires_grad_(False)
    model.satellite_bidirectional.requires_grad_(True)
    model.satellite_memory.requires_grad_(True)
    if config.get("fusion_checkpoint"):
        weights = torch.load(config["fusion_checkpoint"], map_location="cpu", weights_only=True)
        model.satellite_bidirectional.load_state_dict(weights["satellite_bidirectional"], strict=True)
        model.satellite_memory.load_state_dict(weights["satellite_memory"], strict=True)
    return model.to(device).train()


def train(config):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    module_name, function_name = config["data_factory"].split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    model = build_model(config, device)
    parameters = list(model.satellite_bidirectional.parameters()) + list(model.satellite_memory.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.get("learning_rate", 1e-4))
    criterion = LoGeRLoss(lambda_global=config.get("lambda_global", 1.0),
                          lambda_rotation=config.get("lambda_rotation", 0.1),
                          lambda_translation=config.get("lambda_translation", 10.0))
    output_path = Path(config["output_checkpoint"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    window_size = config.get("window_size", 6)
    overlap_size = config.get("overlap_size", 1)
    for epoch in range(config.get("epochs", 1)):
        for batch in factory(config, "train"):
            images = batch["images"].to(device)
            tile = batch["tile"].to(device)
            tile_id = batch["tile_id"].to(device)
            tile_bbox = batch["tile_bbox"].to(device)
            with torch.no_grad():
                satellite_features = model.encode_roma_satellite(tile)
            output = model(images, roma_tile_image=tile, roma_tile_id=tile_id,
                           roma_tile_bbox=tile_bbox,
                           roma_satellite_features=satellite_features,
                           window_size=window_size, overlap_size=overlap_size, no_detach=True)
            targets = {key: batch[key].to(device) for key in ("depth", "intrinsics", "poses")}
            loss = criterion(output, targets)["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.save({"satellite_bidirectional": model.satellite_bidirectional.state_dict(),
                    "satellite_memory": model.satellite_memory.state_dict(), "epoch": epoch + 1},
                   output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    train(config)


if __name__ == "__main__":
    main()

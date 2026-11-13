"""Train one Satellite Region Attention ablation (B or C)."""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from loger.models.pi3 import Pi3
from loger.models.layers.satellite_tiled_features import SatelliteTiledDINO
from train.loss import LoGeRLoss
from train.satellite_tiles import make_loader


def encode_dop_cache(datasets, device, tile_batch_size=4):
    encoder = SatelliteTiledDINO(pretrained=True, tile_batch_size=tile_batch_size).to(device)
    encoder.eval()
    sequences = {str(path): path for dataset in datasets for path in dataset.paths}
    cache = {}
    started = time.time()
    for index, (key, sequence) in enumerate(sequences.items(), 1):
        path = sequence / "dop_view.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        dense = encoder.encode_path(path, device)
        cache[key] = dense.cpu().half()
        print(f"DOP features {index}/{len(sequences)}: {sequence} "
              f"shape={tuple(dense.shape)} elapsed={time.time()-started:.1f}s", flush=True)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


def load_model(config, device):
    model_config = yaml.safe_load(Path(config["model_config"]).read_text())["model"]
    model_config.pop("se3", None)
    model_config.update(
        roma_enabled=False,
        satellite_region_enabled=True,
        satellite_region_use_r2l=bool(config["use_region_to_local"]),
        decoder_gradient_checkpointing=True,
        ttt_gradient_checkpointing=False,
    )
    model = Pi3(**model_config)
    checkpoint = torch.load(config["loger_checkpoint"], map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.to(device).train()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not name.startswith("satellite_region.") for name, _ in trainable):
        raise RuntimeError("Only Satellite Region Attention parameters should be trainable")
    return model, [p for _, p in trainable]


def _sequence(batch):
    value = batch["sequence"]
    return value[0] if isinstance(value, (list, tuple)) else value


def forward(model, batch, cache, device, config):
    sequence = _sequence(batch)
    features = cache[sequence].to(device, non_blocking=True)
    return model(
        batch["images"].to(device, non_blocking=True),
        satellite_region_features=features,
        window_size=int(config["window_size"]),
        overlap_size=int(config["overlap_size"]),
        num_iterations=1,
        no_detach=True,
    )


def batch_loss(prediction, batch, criterion, device):
    target = {key: batch[key].to(device, non_blocking=True)
              for key in ("depth", "intrinsics", "poses")}
    return criterion(prediction, target)


@torch.no_grad()
def evaluate(model, loader, cache, device, config, criterion, max_steps):
    model.eval()
    totals = {key: 0.0 for key in ("total", "local", "pose", "global")}
    count = 0
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction = forward(model, batch, cache, device, config)
            losses = batch_loss(prediction, batch, criterion, device)
        for key in totals:
            totals[key] += float(losses[key])
        count += 1
        if max_steps > 0 and count >= max_steps:
            break
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(config):
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "latest.pt").exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory {output}")
    shutil.copy2(config["config_path"], output / "config.yaml")
    shutil.copy2(config["split_file"], output / "split.json")
    split = json.loads(Path(config["split_file"]).read_text())
    dataset_args = (6, int(config["sample_stride"]), tuple(config["image_size"]))
    train_ds, train_loader = make_loader(
        split["train"], *dataset_args, True, config["satellite_image"], seed)
    val_ds, val_loader = make_loader(
        split["val"], *dataset_args, False, config["satellite_image"], seed)

    device = torch.device(config["device"])
    cache = encode_dop_cache((train_ds, val_ds), device,
                             int(config.get("tile_batch_size", 4)))
    model, parameters = load_model(config, device)
    criterion = LoGeRLoss(float(config["lambda_global"]),
                          float(config["lambda_rotation"]),
                          float(config["lambda_translation"]))
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    writer = SummaryWriter(output / "tensorboard")
    initial_val = evaluate(model, val_loader, cache, device, config, criterion,
                           int(config["val_steps"]))
    best = float(initial_val["total"])
    initial = {"satellite_region": model.satellite_region.state_dict(), "step": 0,
               "val": initial_val, "config": config}
    torch.save(initial, output / "best.pt")
    print(f"Region {config['variant']}: {len(split['train'])} train / {len(split['val'])} val; "
          f"r2l={config['use_region_to_local']} steps={config['steps']} "
          f"trainable={sum(p.numel() for p in parameters):,}; initial_val={initial_val}",
          flush=True)

    iterator = iter(train_loader)
    started = time.time()
    for step in range(1, int(config["steps"]) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction = forward(model, batch, cache, device, config)
            losses = batch_loss(prediction, batch, criterion, device)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        for key in ("total", "local", "pose", "global"):
            writer.add_scalar(f"train/{key}", losses[key].detach(), step)
        writer.add_scalar("train/grad_norm", grad_norm, step)
        for index, gate in enumerate(model.satellite_region.joint_gates):
            writer.add_scalar(f"train/joint_gate_{index}", gate.detach(), step)
        if device.type == "cuda":
            writer.add_scalar("train/gpu_peak_gb",
                              torch.cuda.max_memory_allocated(device) / 1024**3, step)

        if step % int(config["validate_every"]) == 0 or step == int(config["steps"]):
            val = evaluate(model, val_loader, cache, device, config, criterion,
                           int(config["val_steps"]))
            for key, value in val.items():
                writer.add_scalar(f"val/{key}", value, step)
            state = {"satellite_region": model.satellite_region.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": step,
                     "val": val, "config": config}
            torch.save(state, output / "latest.pt")
            if val["total"] < best:
                best = val["total"]
                shutil.copy2(output / "latest.pt", output / "best.pt")
            print(f"{config['variant']} step={step}/{config['steps']} "
                  f"train={float(losses['total']):.5f} val={val['total']:.5f} "
                  f"local={val['local']:.5f} pose={val['pose']:.5f} "
                  f"grad={float(grad_norm):.4g} "
                  f"peak={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB",
                  flush=True)
            model.train()

    summary = {"variant": config["variant"], "initial_val": initial_val,
               "best_val_loss": best, "steps": int(config["steps"]),
               "elapsed_seconds": time.time() - started,
               "base_checkpoint": config["loger_checkpoint"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    writer.close()
    train_ds.close()
    val_ds.close()
    print(f"Finished Region {config['variant']}: {summary}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    config = yaml.safe_load(path.read_text())
    config["config_path"] = str(path)
    train(config)


if __name__ == "__main__":
    main()

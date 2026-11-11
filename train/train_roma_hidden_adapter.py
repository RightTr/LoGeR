"""Train a small LoGeR adapter on frozen MatchAnything RoMa hidden tokens."""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.loss import LoGeRLoss
from train.satellite_tiles import build_model, evaluate_loss, make_loader, move_targets, forward_batch


def train(config):
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("best.pt", "latest.pt")):
        raise FileExistsError(f"Refusing to overwrite existing run: {out}")
    shutil.copy2(config["config_path"], out / "config.yaml")
    split_path = Path(config["split_file"])
    shutil.copy2(split_path, out / "split.json")
    split = json.loads(split_path.read_text())
    size = tuple(config["image_size"])
    train_ds, train_loader = make_loader(
        split["train"], 6, config["sample_stride"], size, True,
        config["satellite_image"], seed)
    val_ds, val_loader = make_loader(
        split["val"], 6, config["sample_stride"], size, False,
        config["satellite_image"], seed)

    device = torch.device(config["device"])
    model = build_model(config, device)
    if model.ttt_layers is None:
        raise RuntimeError("RoMa Hidden training requires LoGeR TTT to be enabled")
    model.requires_grad_(False)
    model.roma_loger_fusion.requires_grad_(True)
    params = list(model.roma_loger_fusion.parameters())
    trainable = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if trainable != [f"roma_loger_fusion.{name}"
                     for name, _ in model.roma_loger_fusion.named_parameters()]:
        raise RuntimeError(f"Unexpected trainable modules: {trainable}")

    criterion = LoGeRLoss(config["lambda_global"], config["lambda_rotation"],
                          config["lambda_translation"])
    cache = {}
    writer = SummaryWriter(out / "tensorboard")
    print(f"RoMa Hidden Adapter: {len(split['train'])} train / "
          f"{len(split['val'])} validation tiles; TTT=ON; "
          f"trainable={sum(p.numel() for p in params):,}; device={device}",
          flush=True)

    baseline = evaluate_loss(model, val_loader, device, cache, criterion,
                             config, "off", int(config["val_steps"]))
    initial = evaluate_loss(model, val_loader, device, cache, criterion,
                            config, "hidden", int(config["val_steps"]))
    print(f"Validation baseline={baseline}; zero-init hidden={initial}", flush=True)
    writer.add_scalar("baseline/val_total", baseline["total"], 0)
    writer.add_scalar("initial_hidden/val_total", initial["total"], 0)

    optimizer = torch.optim.AdamW(
        params, lr=float(config["adapter_lr"]),
        weight_decay=float(config["weight_decay"]))
    best = float(initial["total"])
    torch.save({"roma_loger_fusion": model.roma_loger_fusion.state_dict(),
                "step": 0, "phase": "initial_zero_adapter",
                "val_loss": best, "config": config}, out / "best.pt")
    loader_iter = iter(train_loader)
    started = time.time()
    steps = int(config["single_steps"])
    for step in range(1, steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction = forward_batch(model, batch, device, cache, config,
                                       fusion_mode="hidden")
            losses = criterion(prediction, move_targets(batch, device))
        loss = losses["total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        output_grad = model.roma_loger_fusion.output.weight.grad
        if output_grad is None or not torch.isfinite(output_grad).all():
            raise RuntimeError("RoMa Hidden Adapter output did not receive finite gradients")
        if step == 1 and output_grad.norm().item() == 0:
            raise RuntimeError("RoMa Hidden Adapter output gradient is zero")
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        writer.add_scalar("adapter/train_total", loss.detach(), step)
        writer.add_scalar("adapter/train_local", losses["local"].detach(), step)
        writer.add_scalar("adapter/train_pose", losses["pose"].detach(), step)
        writer.add_scalar("adapter/grad_norm", grad, step)
        writer.add_scalar("adapter/delta_ratio",
                          prediction["roma_hidden_fusion_delta_ratio"], step)
        writer.add_scalar("adapter/gpu_peak_gb",
                          torch.cuda.max_memory_allocated(device) / 1024**3
                          if device.type == "cuda" else 0., step)

        if step % int(config["validate_every"]) == 0 or step == steps:
            val = evaluate_loss(model, val_loader, device, cache, criterion,
                                config, "hidden", int(config["val_steps"]))
            writer.add_scalar("adapter/val_total", val["total"], step)
            writer.add_scalar("adapter/val_local", val["local"], step)
            writer.add_scalar("adapter/val_pose", val["pose"], step)
            checkpoint = {
                "roma_loger_fusion": model.roma_loger_fusion.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step,
                "phase": "roma_hidden_adapter", "val_loss": val["total"],
                "config": config,
            }
            torch.save(checkpoint, out / "latest.pt")
            if val["total"] < best:
                best = val["total"]
                shutil.copy2(out / "latest.pt", out / "best.pt")
            print(f"hidden step={step}/{steps} train={float(loss):.5f} "
                  f"val={val['total']:.5f} local={val['local']:.5f} "
                  f"pose={val['pose']:.5f} grad={float(grad):.4g} "
                  f"delta={float(prediction['roma_hidden_fusion_delta_ratio']):.4g} "
                  f"peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GB",
                  flush=True)
            model.train()

    summary = {"baseline": baseline, "zero_init_hidden": initial,
               "steps": steps, "best_val_loss": best,
               "elapsed_seconds": time.time() - started, "ttt_enabled": True,
               "fusion_mode": "roma_hidden"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    writer.close()
    train_ds.close()
    val_ds.close()
    print(f"Finished RoMa Hidden training: {summary}; outputs={out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roma_hidden_adapter_tiles.yaml")
    args = parser.parse_args()
    path = Path(args.config)
    config = yaml.safe_load(path.read_text())
    config["config_path"] = str(path)
    train(config)


if __name__ == "__main__":
    main()

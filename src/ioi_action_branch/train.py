"""Single-GPU or torchrun DDP training over cached Wan latents."""

import argparse
import contextlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import load_config
from .data import LatentDataset, collate_records
from .flow import flow_matching_loss
from .model import create_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_multiplier(step, total, warmup, floor):
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def precision_context(device, precision):
    if precision == "float32":
        return contextlib.nullcontext()
    if precision != "bfloat16" or device.type != "cuda":
        raise ValueError("bfloat16 training requires CUDA; use float32 for CPU smoke tests.")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16.")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def save_training_checkpoint(path, model, optimizer, scheduler, config, epoch, step, backbone):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": 1,
            "config": config,
            "adapter": model.adapter_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed_epochs": epoch,
            "global_step": step,
            "backbone_reference": str(backbone),
        },
        temporary,
    )
    temporary.replace(path)


def train(args):
    cfg = load_config(args.config)
    tc = cfg["training"]
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed_everything(tc["seed"])
    model = create_model(cfg, pretrained=args.backbone).to(device)
    model.train()
    dataset = LatentDataset(args.manifest, cfg)
    if tc["batch_size"] < 1 or tc["gradient_accumulation"] < 1 or tc["epochs"] < 1:
        raise ValueError("batch_size, gradient_accumulation, and epochs must be positive.")
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=tc["seed"]
    )
    loader = DataLoader(
        dataset,
        batch_size=tc["batch_size"],
        sampler=sampler,
        num_workers=tc.get("num_workers", 0),
        collate_fn=collate_records,
        pin_memory=device.type == "cuda",
    )
    accumulation = tc["gradient_accumulation"]
    total_updates = math.ceil(len(loader) / accumulation) * tc["epochs"]
    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=tc["learning_rate"],
        betas=tuple(tc["betas"]),
        weight_decay=tc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(
            step, total_updates, tc["warmup_steps"], tc["min_learning_rate"] / tc["learning_rate"]
        ),
    )
    start_epoch, global_step = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["config"] != cfg:
            raise ValueError("Resume config must match the saved config exactly.")
        if Path(state["backbone_reference"]).resolve() != Path(args.backbone).resolve():
            raise ValueError("Resume must use the same frozen backbone path.")
        model.load_adapter_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, global_step = state["completed_epochs"], state["global_step"]
    runner = (
        DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
        if world > 1
        else model
    )
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        report = {
            "parameters": model.parameter_counts(),
            "world_size": world,
            "effective_batch": tc["batch_size"] * world * accumulation,
            "updates": total_updates,
        }
        (output / "run_info.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    for epoch in range(start_epoch, tc["epochs"]):
        sampler.set_epoch(epoch)
        # Epoch-boundary resume reproduces data order and rank-local noise draws.
        seed_everything(tc["seed"] + epoch * world + rank)
        generator = torch.Generator(device=device).manual_seed(tc["seed"] + epoch * world + rank)
        optimizer.zero_grad(set_to_none=True)
        window_loss = torch.zeros((), device=device)
        for index, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            window_start = index // accumulation * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            update = index - window_start + 1 == window_size
            sync = runner.no_sync() if world > 1 and not update else contextlib.nullcontext()
            with sync:
                with precision_context(device, tc["precision"]):
                    loss = flow_matching_loss(runner, batch, generator)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={index}.")
                (loss / window_size).backward()
            window_loss += loss.detach() / window_size
            if update:
                torch.nn.utils.clip_grad_norm_(
                    list(model.trainable_parameters()), tc["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if world > 1:
                    dist.all_reduce(window_loss)
                    window_loss /= world
                if rank == 0 and global_step % tc.get("log_every", 10) == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": global_step,
                                "loss": window_loss.item(),
                                "lr": optimizer.param_groups[0]["lr"],
                            }
                        ),
                        flush=True,
                    )
                window_loss.zero_()
        if rank == 0:
            save_training_checkpoint(
                output / "last.pt",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch + 1,
                global_step,
                args.backbone,
            )
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wan2.1_1.3b.json")
    parser.add_argument(
        "--backbone", required=True, help="Local official Wan2.1-T2V-1.3B directory."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    train(parser.parse_args())


if __name__ == "__main__":
    main()

"""Train the byte-level GPT model from randomly initialized weights."""

import argparse
import glob
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from config import Config, DEFAULT_CONFIG
from dataset import TokenDataset
from model import GPTModel, describe_model


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(args: argparse.Namespace) -> Config:
    config = Config.from_dict(DEFAULT_CONFIG.to_dict())
    for key in (
        "dataset_path", "batch_size", "learning_rate", "training_steps",
        "eval_interval", "eval_steps", "checkpoint_interval", "seed",
    ):
        value = getattr(args, key, None)
        if value is not None:
            setattr(config, key, value)
    return config


def latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt")))
    return paths[-1] if paths else None


@torch.no_grad()
def estimate_validation_loss(
    model: GPTModel,
    dataset: TokenDataset,
    config: Config,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(config.eval_steps):
        x, y = dataset.batch(config.batch_size, device)
        with autocast_context(device):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    # A disabled context keeps the training code identical on CPU and CUDA.
    return torch.autocast(device_type="cpu", enabled=False)


def load_checkpoint(
    path: str,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, float, Dict]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (
        int(checkpoint.get("step", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
        checkpoint,
    )


def save_checkpoint(
    path: str,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    config: Config,
    step: int,
    train_loss: float,
    val_loss: Optional[float],
    best_val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    temporary_path = path + ".partial"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="resume from the newest periodic checkpoint")
    parser.add_argument("--reset", action="store_true", help="ignore existing checkpoints and start at step 0")
    parser.add_argument("--dataset-path", dest="dataset_path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = make_config(args)
    if config.training_steps < 1:
        raise ValueError("training_steps must be positive")
    seed_everything(config.seed)
    device = choose_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    train_data = TokenDataset(
        config.dataset_path, config.context_length, config.validation_split, "train", config.seed
    )
    val_data = TokenDataset(
        config.dataset_path, config.context_length, config.validation_split, "val", config.seed
    )
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(describe_model(model))
    print(f"Train windows: {len(train_data):,} | Validation windows: {len(val_data):,}")

    start_step = 0
    best_val_loss = float("inf")
    resume_path = latest_checkpoint(config.checkpoint_dir) if args.resume and not args.reset else None
    if resume_path:
        start_step, best_val_loss, old_checkpoint = load_checkpoint(
            resume_path, model, optimizer, device
        )
        print(
            f"Resumed {resume_path} at step {start_step:,}; "
            f"best validation loss {best_val_loss:.4f}"
        )
    elif args.resume:
        print("No periodic checkpoint found; starting from randomly initialized weights.")
    if args.reset:
        print("Reset requested: starting from randomly initialized weights.")

    model.train()
    run_start = time.perf_counter()
    last_step_time = run_start
    latest_train_loss = float("nan")
    latest_val_loss: Optional[float] = None
    for step in range(start_step + 1, config.training_steps + 1):
        x, y = train_data.batch(config.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        latest_train_loss = float(loss.item())

        should_evaluate = step == 1 or step % config.eval_interval == 0 or step == config.training_steps
        if should_evaluate:
            latest_val_loss = estimate_validation_loss(model, val_data, config, device)
            if latest_val_loss < best_val_loss:
                best_val_loss = latest_val_loss
                save_checkpoint(
                    config.best_checkpoint, model, optimizer, config, step,
                    latest_train_loss, latest_val_loss, best_val_loss
                )
                best_marker = " [new best]"
            else:
                best_marker = ""
        else:
            best_marker = ""

        if should_evaluate or step % max(1, config.eval_interval // 5) == 0:
            elapsed = max(time.perf_counter() - run_start, 1e-9)
            tokens_processed = (step - start_step) * config.batch_size * config.context_length
            tokens_per_second = tokens_processed / elapsed
            remaining = max(config.training_steps - step, 0)
            seconds_remaining = remaining * elapsed / max(step - start_step, 1)
            gpu_suffix = ""
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                gpu_suffix = f" | GPU memory {allocated:.2f} GB"
            val_text = f" | val {latest_val_loss:.4f}" if latest_val_loss is not None else ""
            print(
                f"step {step:>7,}/{config.training_steps:,} | train {latest_train_loss:.4f}"
                f"{val_text} | {tokens_per_second:,.0f} tok/s"
                f" | ETA {seconds_remaining / 60:.1f} min{gpu_suffix}{best_marker}",
                flush=True,
            )

        if step % config.checkpoint_interval == 0 or step == config.training_steps:
            path = os.path.join(config.checkpoint_dir, f"checkpoint_{step:07d}.pt")
            save_checkpoint(
                path, model, optimizer, config, step, latest_train_loss,
                latest_val_loss, best_val_loss
            )
            print(f"Saved checkpoint: {path}", flush=True)

    print(f"Training complete. Best model: {config.best_checkpoint}")


if __name__ == "__main__":
    main()
"""Train the local decoder-only language model from randomly initialized weights."""

import argparse
import glob
import json
import math
import os
import random
import time
from contextlib import nullcontext
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


def load_dataset_manifest(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset manifest {path!r} does not exist. "
            "Run prepare_dataset.py before training."
        )
    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("format") != "document-split-v2":
        raise ValueError(
            f"Dataset manifest {path!r} is a legacy token-offset dataset. "
            "Run prepare_dataset.py to create train_tokens.bin, "
            "validation_tokens.bin, and test_tokens.bin before training."
        )
    for split in ("train", "validation", "test"):
        if split not in manifest.get("splits", {}):
            raise ValueError(f"Dataset manifest is missing the {split!r} split.")
    if manifest.get("tokenizer", {}).get("training_source") != "train_documents_only":
        raise ValueError("Dataset tokenizer provenance is not train-only.")
    return manifest


def make_config(args: argparse.Namespace) -> Config:
    config = Config.from_dict(DEFAULT_CONFIG.to_dict())
    for key in (
        "dataset_manifest_path",
        "tokenizer_path",
        "context_length",
        "embedding_dim",
        "num_layers",
        "num_heads",
        "feed_forward_dim",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "min_learning_rate",
        "warmup_steps",
        "training_steps",
        "eval_interval",
        "eval_steps",
        "checkpoint_interval",
        "seed",
    ):
        value = getattr(args, key, None)
        if value is not None:
            setattr(config, key, value)
    manifest = load_dataset_manifest(config.dataset_manifest_path)
    tokenizer_metadata = manifest["tokenizer"]
    preprocessing_metadata = manifest["preprocessing"]
    requested_tokenizer_path = getattr(args, "tokenizer_path", None)
    manifest_tokenizer_path = tokenizer_metadata["path"]
    if requested_tokenizer_path and requested_tokenizer_path != manifest_tokenizer_path:
        raise ValueError(
            "The requested tokenizer does not match the dataset manifest. "
            "Regenerate the dataset or use its tokenizer."
        )
    config.dataset_dtype = preprocessing_metadata["dtype"]
    config.vocab_size = int(tokenizer_metadata["vocab_size"])
    config.tokenizer_path = manifest_tokenizer_path
    return config


def latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt")))
    return paths[-1] if paths else None


def amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device: torch.device):
    dtype = amp_dtype(device)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


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


def learning_rate_at(step: int, config: Config) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(
        config.training_steps - config.warmup_steps, 1
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (
        config.learning_rate - config.min_learning_rate
    ) * cosine


def load_checkpoint(
    path: str,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config,
) -> Tuple[int, float, Dict]:
    # Checkpoints are generated locally by this project and include optimizer
    # and RNG state, so they intentionally use the full trusted serialization
    # format introduced by PyTorch 2.6.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("dataset_manifest_path") != config.dataset_manifest_path:
        raise ValueError(
            f"Checkpoint {path!r} was created with the legacy or a different "
            "dataset pipeline. Use --reset to train from the new document-split "
            "dataset instead of silently mixing data contracts."
        )
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
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    temporary_path = path + ".partial"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="resume from the newest periodic checkpoint")
    parser.add_argument("--reset", action="store_true", help="ignore existing checkpoints and start at step 0")
    parser.add_argument("--dataset-manifest-path", dest="dataset_manifest_path")
    parser.add_argument("--tokenizer-path", dest="tokenizer_path")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--feed-forward-dim", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = make_config(args)
    manifest = load_dataset_manifest(config.dataset_manifest_path)
    if config.training_steps < 1 or config.batch_size < 1:
        raise ValueError("training_steps and batch_size must be positive")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")

    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device()
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Device: {device}")

    train_data = TokenDataset(
        manifest["splits"]["train"]["path"],
        config.context_length,
        config.seed,
        config.dataset_dtype,
    )
    val_data = TokenDataset(
        manifest["splits"]["validation"]["path"],
        config.context_length,
        config.seed + 1,
        config.dataset_dtype,
    )
    test_data = TokenDataset(
        manifest["splits"]["test"]["path"],
        config.context_length,
        config.seed + 2,
        config.dataset_dtype,
    )
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    use_scaler = device.type == "cuda" and amp_dtype(device) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(describe_model(model))
    print(
        f"Train windows: {len(train_data):,} | Validation windows: {len(val_data):,} | "
        f"Test windows: {len(test_data):,} | "
        f"Effective batch: {config.batch_size * config.gradient_accumulation_steps}"
    )

    start_step = 0
    best_val_loss = float("inf")
    resume_path = latest_checkpoint(config.checkpoint_dir) if args.resume and not args.reset else None
    if resume_path:
        start_step, best_val_loss, old_checkpoint = load_checkpoint(
            resume_path, model, optimizer, device, config
        )
        if "torch_rng_state" in old_checkpoint:
            torch.set_rng_state(old_checkpoint["torch_rng_state"])
            np.random.set_state(old_checkpoint["numpy_rng_state"])
            random.setstate(old_checkpoint["python_rng_state"])
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
    latest_train_loss = float("nan")
    latest_val_loss: Optional[float] = None
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, config.training_steps + 1):
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = train_data.batch(config.batch_size, device)
            with autocast_context(device):
                _, loss = model(x, y)
                scaled_loss = loss / config.gradient_accumulation_steps
            accumulated_loss += float(loss.item())
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if use_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        current_lr = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        latest_train_loss = accumulated_loss / config.gradient_accumulation_steps

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
            tokens_processed = (
                step - start_step
            ) * config.batch_size * config.context_length * config.gradient_accumulation_steps
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
                f"{val_text} | lr {current_lr:.2e} | {tokens_per_second:,.0f} tok/s"
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

    test_loss = estimate_validation_loss(model, test_data, config, device)
    print(
        f"Final test loss: {test_loss:.4f} | "
        f"test perplexity: {math.exp(min(test_loss, 20.0)):.2f}"
    )
    print(f"Training complete. Best model: {config.best_checkpoint}")


if __name__ == "__main__":
    main()
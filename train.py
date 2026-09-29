"""Train the local decoder-only language model from randomly initialized weights."""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from config import Config, DEFAULT_CONFIG
from dataset import TokenDataset
from evaluation import evaluate_fixed_token_loss, load_fixed_evaluation_set
from model import GPTModel, describe_model
from tokenizer import load_tokenizer, tokenizer_fingerprint

CHECKPOINT_FORMAT = "training-checkpoint-v2"
SAFE_RESUME_OVERRIDES = {"training_steps"}


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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_identity(path: str, manifest: Dict) -> Dict[str, Any]:
    return {
        "manifest_sha256": sha256_file(path),
        "format": manifest["format"],
        "source_content_hash": manifest["source"]["content_hash"],
        "split_hashes": {
            name: manifest["splits"][name]["sha256"]
            for name in ("train", "validation", "test")
        },
    }


def make_config(args: argparse.Namespace) -> Config:
    config = Config.from_dict(DEFAULT_CONFIG.to_dict())
    for key in (
        "dataset_manifest_path",
        "tokenizer_path",
        "validation_evaluation_path",
        "test_evaluation_path",
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
    tokenizer = load_tokenizer(manifest_tokenizer_path)
    actual_tokenizer_hash = tokenizer_fingerprint(manifest_tokenizer_path)
    if tokenizer_metadata.get("sha256") != actual_tokenizer_hash:
        raise ValueError("Tokenizer fingerprint does not match the dataset manifest.")
    if tokenizer_metadata.get("vocab_size") != tokenizer.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match the dataset manifest.")
    if tokenizer_metadata.get("eos_token_id") != tokenizer.eos_token_id:
        raise ValueError("Tokenizer EOS ID does not match the dataset manifest.")
    if tokenizer_metadata.get("pad_token_id") != tokenizer.pad_token_id:
        raise ValueError("Tokenizer PAD ID does not match the dataset manifest.")
    config.dataset_dtype = preprocessing_metadata["dtype"]
    config.dataset_identity = dataset_identity(config.dataset_manifest_path, manifest)
    config.vocab_size = tokenizer.vocab_size
    config.tokenizer_path = manifest_tokenizer_path
    config.tokenizer_sha256 = actual_tokenizer_hash
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    return config


def latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt")))
    return paths[-1] if paths else None


def amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def amp_dtype_name(device: torch.device) -> str:
    dtype = amp_dtype(device)
    return "none" if dtype is None else str(dtype).split(".")[-1]


def autocast_context(device: torch.device, dtype_name: Optional[str] = None):
    dtype = amp_dtype(device) if dtype_name is None else {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name)
    if dtype_name is not None and dtype_name not in {"none", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported AMP dtype {dtype_name!r}.")
    if dtype is not None and device.type != "cuda":
        raise ValueError(
            f"Checkpoint requests AMP dtype {dtype_name!r}, but the current device "
            f"is {device.type!r}."
        )
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def estimate_validation_loss(
    model: GPTModel,
    evaluation_set: Dict[str, Any],
    config: Config,
    device: torch.device,
) -> float:
    return float(
        evaluate_fixed_token_loss(
            model,
            evaluation_set,
            config.batch_size,
            device,
            config.amp_dtype,
        )["cross_entropy"]
    )


def learning_rate_at(
    step: int, config: Config, total_steps: Optional[int] = None
) -> float:
    schedule_steps = config.training_steps if total_steps is None else total_steps
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(
        schedule_steps - config.warmup_steps, 1
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (
        config.learning_rate - config.min_learning_rate
    ) * cosine


class LearningRateScheduler:
    """Serializable warmup/cosine scheduler used by the training loop."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: Config) -> None:
        self.optimizer = optimizer
        self.schedule_config = {
            "name": "warmup-cosine",
            "learning_rate": config.learning_rate,
            "min_learning_rate": config.min_learning_rate,
            "warmup_steps": config.warmup_steps,
            "total_steps": config.training_steps,
        }
        self.step_num = 0
        self.last_lr = [float(group["lr"]) for group in optimizer.param_groups]

    def step(self, step: int) -> float:
        if step != self.step_num + 1:
            raise ValueError(
                f"Learning-rate scheduler expected step {self.step_num + 1}, "
                f"received {step}."
            )
        self.step_num = step
        learning_rate = learning_rate_at(
            step,
            Config(
                learning_rate=self.schedule_config["learning_rate"],
                min_learning_rate=self.schedule_config["min_learning_rate"],
                warmup_steps=self.schedule_config["warmup_steps"],
                training_steps=self.schedule_config["total_steps"],
            ),
            total_steps=self.schedule_config["total_steps"],
        )
        self.last_lr = [learning_rate for _ in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.schedule_config["name"],
            "config": dict(self.schedule_config),
            "step": self.step_num,
            "last_lr": list(self.last_lr),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        required = {"name", "config", "step", "last_lr"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(
                f"Checkpoint scheduler state is missing required fields: {missing}."
            )
        if state["name"] != self.schedule_config["name"]:
            raise ValueError(
                f"Unsupported checkpoint scheduler {state['name']!r}; "
                f"expected {self.schedule_config['name']!r}."
            )
        saved_config = state["config"]
        for field_name in ("learning_rate", "min_learning_rate", "warmup_steps"):
            if saved_config.get(field_name) != self.schedule_config[field_name]:
                raise ValueError(
                    f"Checkpoint scheduler configuration does not match "
                    f"current configuration ({field_name})."
                )
        self.schedule_config = dict(saved_config)
        self.step_num = int(state["step"])
        self.last_lr = [float(value) for value in state["last_lr"]]
        if len(self.last_lr) != len(self.optimizer.param_groups):
            raise ValueError("Checkpoint scheduler parameter-group count does not match.")
        for group, learning_rate in zip(self.optimizer.param_groups, self.last_lr):
            group["lr"] = learning_rate


def load_checkpoint(
    path: str,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: LearningRateScheduler,
    train_data: TokenDataset,
    validation_data: TokenDataset,
    test_data: TokenDataset,
    device: torch.device,
    config: Config,
) -> Tuple[int, float, Dict]:
    # Checkpoints are generated locally by this project and include optimizer
    # and RNG state, so they intentionally use the full trusted serialization
    # format introduced by PyTorch 2.6.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required_keys = {
        "checkpoint_format",
        "model_state",
        "optimizer_state",
        "scaler_state",
        "scheduler_state",
        "step",
        "best_val_loss",
        "config",
        "torch_rng_state",
        "numpy_rng_state",
        "python_rng_state",
        "train_dataset_rng_state",
        "validation_dataset_rng_state",
        "test_dataset_rng_state",
    }
    missing_keys = sorted(required_keys - set(checkpoint))
    if missing_keys:
        raise ValueError(
            f"Checkpoint {path!r} is incomplete; missing fields: {missing_keys}."
        )
    if checkpoint["checkpoint_format"] != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint {path!r} has unsupported format "
            f"{checkpoint['checkpoint_format']!r}; expected {CHECKPOINT_FORMAT!r}."
        )
    checkpoint_config = checkpoint.get("config", {})
    current_config = config.to_dict()
    missing_config = sorted(set(current_config) - set(checkpoint_config))
    if missing_config:
        raise ValueError(
            f"Checkpoint {path!r} has incomplete configuration; missing fields: "
            f"{missing_config}."
        )
    for field_name, current_value in current_config.items():
        saved_value = checkpoint_config[field_name]
        if field_name in SAFE_RESUME_OVERRIDES:
            if current_value < saved_value:
                raise ValueError(
                    "Cannot resume with fewer training steps than the checkpoint "
                    f"({current_value} < {saved_value}). Only extending training_steps "
                    "is supported."
                )
            continue
        if saved_value != current_value:
            raise ValueError(
                f"Checkpoint configuration mismatch for {field_name}: "
                f"checkpoint={saved_value!r}, current={current_value!r}. "
                "Only extending training_steps is supported when resuming."
            )
    step = int(checkpoint["step"])
    if step > config.training_steps:
        raise ValueError(
            f"Checkpoint step {step} exceeds requested training_steps "
            f"{config.training_steps}."
        )
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    train_data.rng.bit_generator.state = checkpoint["train_dataset_rng_state"]
    validation_data.rng.bit_generator.state = checkpoint["validation_dataset_rng_state"]
    test_data.rng.bit_generator.state = checkpoint["test_dataset_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    return (
        step,
        float(checkpoint["best_val_loss"]),
        checkpoint,
    )


def save_checkpoint(
    path: str,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: LearningRateScheduler,
    train_data: TokenDataset,
    validation_data: TokenDataset,
    test_data: TokenDataset,
    config: Config,
    step: int,
    train_loss: float,
    val_loss: Optional[float],
    best_val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "train_dataset_rng_state": train_data.rng.bit_generator.state,
        "validation_dataset_rng_state": validation_data.rng.bit_generator.state,
        "test_dataset_rng_state": test_data.rng.bit_generator.state,
    }
    temporary_path = path + ".partial"
    try:
        with open(temporary_path, "wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="resume from the newest periodic checkpoint")
    parser.add_argument("--reset", action="store_true", help="ignore existing checkpoints and start at step 0")
    parser.add_argument("--dataset-manifest-path", dest="dataset_manifest_path")
    parser.add_argument("--tokenizer-path", dest="tokenizer_path")
    parser.add_argument("--validation-evaluation-path")
    parser.add_argument("--test-evaluation-path")
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
    validation_evaluation_set = load_fixed_evaluation_set(
        config.validation_evaluation_path,
        config.dataset_manifest_path,
        "validation",
        config.context_length,
    )
    test_evaluation_set = load_fixed_evaluation_set(
        config.test_evaluation_path,
        config.dataset_manifest_path,
        "test",
        config.context_length,
    )
    if config.training_steps < 1 or config.batch_size < 1:
        raise ValueError("training_steps and batch_size must be positive")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")

    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device()
    config.amp_dtype = amp_dtype_name(device)
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
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    use_scaler = device.type == "cuda" and amp_dtype(device) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    scheduler = LearningRateScheduler(optimizer, config)
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
            resume_path,
            model,
            optimizer,
            scaler,
            scheduler,
            train_data,
            val_data,
            test_data,
            device,
            config,
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
    latest_train_loss = float("nan")
    latest_val_loss: Optional[float] = None
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, config.training_steps + 1):
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = train_data.batch(config.batch_size, device)
            with autocast_context(device, config.amp_dtype):
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
        current_lr = scheduler.step(step)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        latest_train_loss = accumulated_loss / config.gradient_accumulation_steps

        should_evaluate = step == 1 or step % config.eval_interval == 0 or step == config.training_steps
        if should_evaluate:
            latest_val_loss = estimate_validation_loss(
                model, validation_evaluation_set, config, device
            )
            if latest_val_loss < best_val_loss:
                best_val_loss = latest_val_loss
                save_checkpoint(
                    config.best_checkpoint,
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    train_data,
                    val_data,
                    test_data,
                    config,
                    step,
                    latest_train_loss,
                    latest_val_loss,
                    best_val_loss,
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
                path,
                model,
                optimizer,
                scaler,
                scheduler,
                train_data,
                val_data,
                test_data,
                config,
                step,
                latest_train_loss,
                latest_val_loss,
                best_val_loss,
            )
            print(f"Saved checkpoint: {path}", flush=True)

    test_loss = estimate_validation_loss(
        model, test_evaluation_set, config, device
    )
    print(
        f"Final test loss: {test_loss:.4f} | "
        f"test perplexity: {math.exp(min(test_loss, 20.0)):.2f}"
    )
    print(f"Training complete. Best model: {config.best_checkpoint}")


if __name__ == "__main__":
    main()
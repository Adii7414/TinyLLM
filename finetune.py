"""Instruction fine-tuning from a pretrained local subword GPT checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn import functional as F

from config import Config
from instruction_dataset import InstructionDataset, instruction_dataset_fingerprint
from model import GPTModel, describe_model
from tokenizer import load_tokenizer, tokenizer_fingerprint
from train import (
    LearningRateScheduler,
    amp_dtype,
    amp_dtype_name,
    autocast_context,
    perplexity_from_loss,
    seed_everything,
)


INSTRUCTION_CHECKPOINT_FORMAT = "instruction-finetuning-v1"
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_MIN_LEARNING_RATE = 5e-6


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pretrained_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pretrained checkpoint {path!r} does not exist. "
            "Run train.py before instruction fine-tuning."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    for key in ("model_state", "config"):
        if key not in checkpoint:
            raise ValueError(f"Pretrained checkpoint is missing {key!r}.")
    return checkpoint


def load_instruction_model(
    checkpoint: Dict[str, Any],
    instruction_tokenizer_path: str,
    device: torch.device,
) -> Tuple[GPTModel, Config]:
    base_config = Config.from_dict(checkpoint["config"])
    base_tokenizer = load_tokenizer(base_config.tokenizer_path)
    instruction_tokenizer = load_tokenizer(instruction_tokenizer_path)
    if instruction_tokenizer.vocab_size < base_tokenizer.vocab_size:
        raise ValueError("Instruction tokenizer cannot shrink the pretrained vocabulary.")
    if instruction_tokenizer.tokens[: base_tokenizer.vocab_size] != base_tokenizer.tokens:
        raise ValueError(
            "Instruction tokenizer must preserve the pretrained tokenizer prefix."
        )

    config = Config.from_dict(base_config.to_dict())
    config.vocab_size = instruction_tokenizer.vocab_size
    config.tokenizer_path = instruction_tokenizer_path
    config.tokenizer_sha256 = tokenizer_fingerprint(instruction_tokenizer_path)
    config.eos_token_id = instruction_tokenizer.eos_token_id
    config.pad_token_id = instruction_tokenizer.pad_token_id
    model = GPTModel(config).to(device)

    target_state = model.state_dict()
    for name, source_value in checkpoint["model_state"].items():
        if name not in target_state:
            raise ValueError(f"Pretrained checkpoint contains unknown model field {name!r}.")
        target_value = target_state[name]
        if target_value.shape == source_value.shape:
            target_value.copy_(source_value)
        elif name in {"token_embedding.weight", "language_model_head.weight"}:
            if (
                target_value.ndim != source_value.ndim
                or target_value.shape[1:] != source_value.shape[1:]
                or target_value.shape[0] < source_value.shape[0]
            ):
                raise ValueError(
                    f"Cannot expand pretrained parameter {name!r} from "
                    f"{tuple(source_value.shape)} to {tuple(target_value.shape)}."
                )
            target_value[: source_value.shape[0]].copy_(source_value)
        else:
            raise ValueError(
                f"Pretrained parameter {name!r} has incompatible shape "
                f"{tuple(source_value.shape)} versus {tuple(target_value.shape)}."
            )
    return model, config


def resolve_steps(
    dataset: InstructionDataset,
    batch_size: int,
    gradient_accumulation_steps: int,
    target_epochs: float,
    training_steps: int,
) -> Tuple[int, int]:
    if batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive.")
    if target_epochs <= 0:
        raise ValueError("target_epochs must be positive.")
    updates_per_epoch = math.ceil(len(dataset) / (batch_size * gradient_accumulation_steps))
    if training_steps < 1:
        training_steps = max(1, math.ceil(target_epochs * updates_per_epoch))
    return training_steps, updates_per_epoch


@torch.no_grad()
def evaluate_instruction_loss(
    model: GPTModel,
    dataset: InstructionDataset,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
    amp_dtype_name_value: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_labels = 0
    for inputs, targets in dataset.batches(batch_size, device, pad_token_id):
        with autocast_context(device, amp_dtype_name_value):
            logits, _ = model(inputs, targets)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        label_count = int((targets != -100).sum().item())
        total_loss += float(loss_sum.item())
        total_labels += label_count
    if total_labels < 1:
        raise ValueError("Instruction validation set has no assistant labels.")
    return total_loss / total_labels


def save_instruction_checkpoint(
    path: str,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: LearningRateScheduler,
    config: Config,
    step: int,
    train_loss: float,
    validation_loss: float,
    best_validation_loss: float,
    train_data: InstructionDataset,
    validation_data: InstructionDataset,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "checkpoint_format": INSTRUCTION_CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "data": {
            "train_path": train_data.path,
            "train_sha256": instruction_dataset_fingerprint(train_data.path),
            "validation_path": validation_data.path,
            "validation_sha256": instruction_dataset_fingerprint(validation_data.path),
            "final_test_used": False,
        },
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
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer-path", default="instruction_tokenizer.json")
    parser.add_argument("--train-data", default="instruction_train.jsonl")
    parser.add_argument("--validation-data", default="instruction_validation.jsonl")
    parser.add_argument(
        "--output-checkpoint",
        default="checkpoints/instruction_best_model.pt",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--min-learning-rate", type=float, default=DEFAULT_MIN_LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--target-epochs", type=float, default=3.0)
    parser.add_argument("--training-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_pretrained_checkpoint(args.checkpoint, device)
    base_config = Config.from_dict(checkpoint["config"])
    if args.learning_rate >= base_config.learning_rate:
        raise ValueError(
            f"Instruction fine-tuning learning rate {args.learning_rate:g} must be "
            f"lower than pretraining rate {base_config.learning_rate:g}."
        )
    tokenizer = load_tokenizer(args.tokenizer_path)
    model, config = load_instruction_model(checkpoint, args.tokenizer_path, device)
    config.learning_rate = args.learning_rate
    config.min_learning_rate = args.min_learning_rate
    config.warmup_steps = args.warmup_steps
    config.warmup_fraction = args.warmup_fraction
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.eval_interval = args.eval_interval
    config.checkpoint_interval = args.checkpoint_interval
    config.seed = args.seed

    train_data = InstructionDataset(
        args.train_data,
        tokenizer,
        config.context_length,
        seed=args.seed,
    )
    validation_data = InstructionDataset(
        args.validation_data,
        tokenizer,
        config.context_length,
        seed=args.seed + 1,
    )
    config.training_steps, updates_per_epoch = resolve_steps(
        train_data,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.target_epochs,
        args.training_steps,
    )
    if config.warmup_steps < 1:
        config.warmup_steps = max(
            1,
            min(
                config.training_steps - 1,
                round(config.training_steps * config.warmup_fraction),
            ),
        )

    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    config.amp_dtype = amp_dtype_name(device)
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
    print(f"Device: {device}")
    print(describe_model(model))
    print(
        f"Instruction examples: train={len(train_data):,}, "
        f"validation={len(validation_data):,}"
    )
    print(
        f"Assistant labels: train={train_data.supervised_tokens:,}, "
        f"validation={validation_data.supervised_tokens:,}"
    )
    print(
        f"Instruction schedule: {config.training_steps:,} updates, "
        f"{args.target_epochs:.2f} example epochs, "
        f"{updates_per_epoch:,} updates/example epoch"
    )
    print(
        f"Learning rate: {config.learning_rate:.2e} "
        f"(pretraining was {base_config.learning_rate:.2e})"
    )

    best_validation_loss = float("inf")
    latest_train_loss = float("nan")
    latest_validation_loss = float("nan")
    model.train()
    run_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config.training_steps + 1):
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            inputs, targets = train_data.batch(
                config.batch_size,
                device,
                tokenizer.pad_token_id,
            )
            with autocast_context(device, config.amp_dtype):
                _, loss = model(inputs, targets)
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

        should_evaluate = (
            step == 1
            or step % config.eval_interval == 0
            or step == config.training_steps
        )
        best_marker = ""
        if should_evaluate:
            latest_validation_loss = evaluate_instruction_loss(
                model,
                validation_data,
                config.batch_size,
                device,
                tokenizer.pad_token_id,
                config.amp_dtype,
            )
            if latest_validation_loss < best_validation_loss:
                best_validation_loss = latest_validation_loss
                save_instruction_checkpoint(
                    args.output_checkpoint,
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    config,
                    step,
                    latest_train_loss,
                    latest_validation_loss,
                    best_validation_loss,
                    train_data,
                    validation_data,
                )
                best_marker = " [new best]"
            model.train()

        if should_evaluate or step % max(1, config.eval_interval // 5) == 0:
            elapsed = max(time.perf_counter() - run_start, 1e-9)
            tokens_processed = (
                step
                * config.batch_size
                * config.gradient_accumulation_steps
                * config.context_length
            )
            epoch = step / max(updates_per_epoch, 1)
            print(
                f"step {step:>6,}/{config.training_steps:,}"
                f" | epoch {epoch:.3f}"
                f" | tokens {tokens_processed:,}"
                f" | train_loss {latest_train_loss:.4f}"
                f" | val_loss {latest_validation_loss:.4f}"
                f" | train_ppl {perplexity_from_loss(latest_train_loss):.3f}"
                f" | val_ppl {perplexity_from_loss(latest_validation_loss):.3f}"
                f" | lr {current_lr:.2e}"
                f" | {step / elapsed:.2f} updates/s{best_marker}",
                flush=True,
            )

    if not os.path.exists(args.output_checkpoint):
        raise RuntimeError("Fine-tuning completed without saving a best checkpoint.")
    print(
        f"Instruction fine-tuning complete. Best validation loss: "
        f"{best_validation_loss:.4f}. Saved {args.output_checkpoint!r}."
    )


if __name__ == "__main__":
    main()
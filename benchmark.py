"""Measure forward/backward throughput on the current machine."""

import argparse
import time

import torch

from config import Config, DEFAULT_CONFIG
from model import GPTModel, describe_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--checkpoint")
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = Config.from_dict(checkpoint["config"])
    else:
        config = Config.from_dict(DEFAULT_CONFIG.to_dict())
    model = GPTModel(config).to(device)
    if args.checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    model.train(not args.inference_only)
    x = torch.randint(0, config.vocab_size, (args.batch_size, config.context_length), device=device)
    y = torch.randint(0, config.vocab_size, (args.batch_size, config.context_length), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate) if not args.inference_only else None

    for _ in range(2):  # warm up kernels before timing
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, None if args.inference_only else y)
        if not args.inference_only:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(args.steps):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, None if args.inference_only else y)
        if not args.inference_only:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = args.steps * args.batch_size * config.context_length
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")
    print(f"Mode: {'inference' if args.inference_only else 'training'}")
    print(f"Steps: {args.steps}")
    print(f"Steps/second: {args.steps / elapsed:.2f}")
    print(f"Tokens/second: {tokens / elapsed:,.0f}")
    print(f"Estimated time for 1M tokens: {1_000_000 / (tokens / elapsed) / 60:.1f} minutes")
    print(describe_model(model))


if __name__ == "__main__":
    main()
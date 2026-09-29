"""Print architecture and parameter information without training."""

import argparse
import glob
import os

import torch

from config import Config, DEFAULT_CONFIG
from generate import load_tokenizer_for_model
from model import GPTModel, describe_model
from tokenizer import load_tokenizer, tokenizer_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        help="inspect a checkpoint's saved configuration; otherwise inspect defaults",
    )
    args = parser.parse_args()
    device = torch.device("cpu")
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = Config.from_dict(checkpoint["config"])
    else:
        config = Config.from_dict(DEFAULT_CONFIG.to_dict())
        tokenizer = load_tokenizer(config.tokenizer_path)
        config.vocab_size = tokenizer.vocab_size
        config.tokenizer_sha256 = tokenizer_fingerprint(config.tokenizer_path)
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
    model = GPTModel(config)
    print(describe_model(model))
    if args.checkpoint:
        load_tokenizer_for_model(model)
        print(f"Checkpoint: {args.checkpoint}")
        if "step" in checkpoint:
            print(f"Training step: {checkpoint['step']:,}")
    else:
        print("Weights: randomly initialized (configuration inspection only)")


if __name__ == "__main__":
    main()
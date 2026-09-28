"""Generate text from a locally trained checkpoint."""

import argparse
import glob
import os
from typing import Optional

import torch

from config import Config, DEFAULT_CONFIG
from model import GPTModel
from tokenizer import decode, encode


def checkpoint_path(requested: Optional[str]) -> str:
    if requested:
        return requested
    if os.path.exists(DEFAULT_CONFIG.best_checkpoint):
        return DEFAULT_CONFIG.best_checkpoint
    candidates = sorted(glob.glob(os.path.join(DEFAULT_CONFIG.checkpoint_dir, "checkpoint_*.pt")))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("No checkpoint found. Train the model before generating.")


def load_model(path: str, device: torch.device) -> GPTModel:
    checkpoint = torch.load(path, map_location=device)
    config = Config.from_dict(checkpoint["config"])
    model = GPTModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def generate_tokens(
    model: GPTModel,
    prompt_ids,
    max_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50,
):
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    token_ids = list(prompt_ids)
    for _ in range(max_tokens):
        context = token_ids[-model.config.context_length :]
        input_ids = torch.tensor([context], dtype=torch.long, device=next(model.parameters()).device)
        logits, _ = model(input_ids)
        next_logits = logits[0, -1, :]
        if temperature == 0:
            next_token = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[-1]] = -float("inf")
            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probabilities, num_samples=1).item())
        token_ids.append(next_token)
    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="checkpoint path; defaults to best_model.pt")
    parser.add_argument("--prompt", help="prompt text; omit for interactive prompt entry")
    parser.add_argument("--max_tokens", "--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    prompt = args.prompt if args.prompt is not None else input("Prompt: ")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = checkpoint_path(args.checkpoint)
    model = load_model(path, device)
    output_ids = generate_tokens(model, encode(prompt), args.max_tokens, args.temperature, args.top_k)
    print("\n" + decode(output_ids))
    print(f"\n[checkpoint={path}, device={device}]")


if __name__ == "__main__":
    main()
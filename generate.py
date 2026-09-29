"""Generate text from a locally trained checkpoint."""

import argparse
import glob
import os
from typing import Optional

import torch

from config import Config, DEFAULT_CONFIG
from model import GPTModel
from tokenizer import ByteSubwordTokenizer, load_tokenizer, tokenizer_fingerprint


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
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = Config.from_dict(checkpoint["config"])
    model = GPTModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    load_tokenizer_for_model(model)
    return model


def load_tokenizer_for_model(model: GPTModel) -> ByteSubwordTokenizer:
    """Load and verify the tokenizer contract embedded in a checkpoint."""
    tokenizer = load_tokenizer(model.config.tokenizer_path)
    checks = {
        "vocab_size": (tokenizer.vocab_size, model.config.vocab_size),
        "eos_token_id": (tokenizer.eos_token_id, model.config.eos_token_id),
        "pad_token_id": (tokenizer.pad_token_id, model.config.pad_token_id),
        "tokenizer_sha256": (
            tokenizer_fingerprint(model.config.tokenizer_path),
            model.config.tokenizer_sha256,
        ),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"Checkpoint/tokenizer mismatch for {field}: "
                f"checkpoint={expected!r}, tokenizer={actual!r}."
            )
    return tokenizer


@torch.no_grad()
def generate_tokens(
    model: GPTModel,
    prompt_ids,
    max_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.08,
    eos_token_id: Optional[int] = None,
):
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be in the range (0, 1].")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be at least 1.")
    token_ids = list(prompt_ids)
    for _ in range(max_tokens):
        context = token_ids[-model.config.context_length :]
        input_ids = torch.tensor([context], dtype=torch.long, device=next(model.parameters()).device)
        logits, _ = model(input_ids)
        next_logits = logits[0, -1, :].float()
        if repetition_penalty > 1 and token_ids:
            recent = set(token_ids[-model.config.context_length :])
            for token_id in recent:
                if next_logits[token_id] < 0:
                    next_logits[token_id] *= repetition_penalty
                else:
                    next_logits[token_id] /= repetition_penalty
        if temperature == 0:
            next_token = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[-1]] = -float("inf")
            probabilities = torch.softmax(next_logits, dim=-1)
            if top_p < 1:
                sorted_probabilities, sorted_indices = torch.sort(
                    probabilities, descending=True
                )
                cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                remove = cumulative - sorted_probabilities > top_p
                probabilities[sorted_indices[remove]] = 0
                probabilities = probabilities / probabilities.sum()
            next_token = int(torch.multinomial(probabilities, num_samples=1).item())
        token_ids.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="checkpoint path; defaults to best_model.pt")
    parser.add_argument("--prompt", help="prompt text; omit for interactive prompt entry")
    parser.add_argument("--max_tokens", "--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    prompt = args.prompt if args.prompt is not None else input("Prompt: ")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = checkpoint_path(args.checkpoint)
    model = load_model(path, device)
    tokenizer = load_tokenizer_for_model(model)
    prompt_ids = tokenizer.encode(prompt)
    output_ids = generate_tokens(
        model,
        prompt_ids,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        tokenizer.eos_token_id,
    )
    print("\n" + tokenizer.decode(output_ids))
    print(f"\n[checkpoint={path}, device={device}]")


if __name__ == "__main__":
    main()
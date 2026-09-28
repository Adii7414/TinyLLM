"""A tiny local chat loop using the trained model as an autoregressive completer."""

import argparse

import torch

from generate import checkpoint_path, generate_tokens, load_model
from tokenizer import decode, encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--message", help="run one message and exit instead of opening a loop")
    parser.add_argument("--max_tokens", "--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = checkpoint_path(args.checkpoint)
    model = load_model(path, device)
    print("Small local byte-level GPT chat.")
    print("This is an educational model, not ChatGPT; its replies depend entirely on your data.")
    print("Type 'exit' or press Ctrl-D to leave.\n")

    history = []
    while True:
        try:
            message = args.message if args.message is not None else input("You: ")
        except EOFError:
            print()
            break
        if message.strip().lower() in {"exit", "quit"}:
            break
        history.append(f"User: {message}\nAssistant:")
        prompt = "\n".join(history)
        prompt_ids = encode(prompt)
        output_ids = generate_tokens(
            model, prompt_ids, args.max_tokens, args.temperature, args.top_k
        )
        # Slice token IDs, not decoded characters: one Unicode character can
        # occupy multiple UTF-8 byte tokens.
        generated = decode(output_ids[len(prompt_ids) :])
        # Keep a compact transcript so old turns do not crowd out the newest one.
        answer = generated.split("\nUser:", 1)[0].strip()
        print(f"Model: {answer}\n")
        history.append(f" {answer}")
        if args.message is not None:
            break


if __name__ == "__main__":
    main()
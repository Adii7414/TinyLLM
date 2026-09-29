"""A tiny local chat loop using the trained model as an autoregressive completer."""

import argparse

import torch

from generate import checkpoint_path, generate_tokens, load_model, load_tokenizer_for_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--message", help="run one message and exit instead of opening a loop")
    parser.add_argument("--max_tokens", "--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--history-turns", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = checkpoint_path(args.checkpoint)
    model = load_model(path, device)
    tokenizer = load_tokenizer_for_model(model)
    if args.history_turns < 1:
        raise ValueError("history-turns must be positive")
    print("Local subword GPT chat.")
    print("Responses depend entirely on the data used for training and fine-tuning.")
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
        history.append(f"<|user|>\n{message}\n<|assistant|>\n")
        prompt = "\n".join(history)
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
        generated = tokenizer.decode(output_ids[len(prompt_ids) :])
        # Keep a compact transcript so old turns do not crowd out the newest one.
        answer = generated.split("<|user|>", 1)[0].strip()
        print(f"Model: {answer}\n")
        history.append(answer)
        history = history[-2 * args.history_turns :]
        if args.message is not None:
            break


if __name__ == "__main__":
    main()
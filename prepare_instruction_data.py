"""Prepare train/validation conversation files and an instruction tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

from instruction_dataset import (
    InstructionExample,
    instruction_dataset_fingerprint,
    load_instruction_examples,
)
from tokenizer import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    INSTRUCTION_SPECIAL_TOKENS,
    USER_TOKEN,
    load_tokenizer,
    tokenizer_fingerprint,
)


MANIFEST_FORMAT = "instruction-dataset-v1"


def split_examples(
    examples: Iterable[InstructionExample],
    validation_ratio: float,
    split_seed: int,
) -> Dict[str, List[InstructionExample]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    train: List[InstructionExample] = []
    validation: List[InstructionExample] = []
    for example in examples:
        key = (
            f"{split_seed}\0{example.category}\0{example.question}\0{example.answer}"
        ).encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
        (validation if value < validation_ratio else train).append(example)
    if not train or not validation:
        raise ValueError("Instruction split produced an empty train or validation set.")
    return {"train": train, "validation": validation}


def write_jsonl(path: str, examples: Iterable[InstructionExample]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary_path = path + ".partial"
    with open(temporary_path, "w", encoding="utf-8") as file:
        for example in examples:
            json.dump(
                {
                    "category": example.category,
                    "question": example.question,
                    "answer": example.answer,
                },
                file,
                ensure_ascii=False,
            )
            file.write("\n")
    os.replace(temporary_path, path)


def prepare_instruction_data(
    source_path: str,
    base_tokenizer_path: str,
    output_tokenizer_path: str,
    train_output_path: str,
    validation_output_path: str,
    manifest_path: str,
    validation_ratio: float = 0.2,
    split_seed: int = 1337,
) -> Dict:
    if "evaluation/test" in os.path.normpath(source_path):
        raise ValueError("The final test benchmark cannot be used as instruction data.")
    examples = load_instruction_examples(source_path)
    splits = split_examples(examples, validation_ratio, split_seed)
    base_tokenizer = load_tokenizer(base_tokenizer_path)
    instruction_tokenizer = base_tokenizer.add_special_tokens(INSTRUCTION_SPECIAL_TOKENS)
    for name in INSTRUCTION_SPECIAL_TOKENS:
        instruction_tokenizer.special_token_id(name)
    instruction_tokenizer.save(output_tokenizer_path)
    write_jsonl(train_output_path, splits["train"])
    write_jsonl(validation_output_path, splits["validation"])
    manifest = {
        "format": MANIFEST_FORMAT,
        "source": {
            "path": source_path,
            "sha256": instruction_dataset_fingerprint(source_path),
            "example_count": len(examples),
        },
        "split": {
            "seed": split_seed,
            "validation_ratio": validation_ratio,
            "counts": {name: len(values) for name, values in splits.items()},
        },
        "tokenizer": {
            "base_path": base_tokenizer_path,
            "path": output_tokenizer_path,
            "sha256": tokenizer_fingerprint(output_tokenizer_path),
            "base_vocab_size": base_tokenizer.vocab_size,
            "vocab_size": instruction_tokenizer.vocab_size,
            "special_tokens": {
                USER_TOKEN: instruction_tokenizer.special_token_id(USER_TOKEN),
                ASSISTANT_TOKEN: instruction_tokenizer.special_token_id(ASSISTANT_TOKEN),
                END_TOKEN: instruction_tokenizer.special_token_id(END_TOKEN),
            },
        },
        "formatting": (
            f"{USER_TOKEN}\\nquestion\\n{ASSISTANT_TOKEN}\\nanswer\\n{END_TOKEN}"
        ),
        "loss_policy": (
            "Labels are -100 for user prompt and role-prefix positions; "
            "assistant answer tokens and the end marker are supervised."
        ),
        "test_isolation": (
            "The final evaluation benchmark and final TEST split are not used "
            "for instruction data, validation, or checkpoint selection."
        ),
        "splits": {
            "train": {"path": train_output_path},
            "validation": {"path": validation_output_path},
        },
    }
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path + ".partial", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(manifest_path + ".partial", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="instruction_data.jsonl")
    parser.add_argument("--base-tokenizer", default="tokenizer.json")
    parser.add_argument("--tokenizer", default="instruction_tokenizer.json")
    parser.add_argument("--train-output", default="instruction_train.jsonl")
    parser.add_argument("--validation-output", default="instruction_validation.jsonl")
    parser.add_argument("--manifest", default="instruction_manifest.json")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1337)
    args = parser.parse_args()
    manifest = prepare_instruction_data(
        source_path=args.input,
        base_tokenizer_path=args.base_tokenizer,
        output_tokenizer_path=args.tokenizer,
        train_output_path=args.train_output,
        validation_output_path=args.validation_output,
        manifest_path=args.manifest,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
    )
    print(
        f"Prepared instruction examples: train={manifest['split']['counts']['train']}, "
        f"validation={manifest['split']['counts']['validation']}"
    )
    print(f"Instruction tokenizer vocabulary: {manifest['tokenizer']['vocab_size']}")
    print(f"Manifest written to {args.manifest!r}")


if __name__ == "__main__":
    main()
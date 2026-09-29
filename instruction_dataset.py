"""Conversation formatting and assistant-only masked batches for instruction tuning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch

from tokenizer import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    USER_TOKEN,
    ByteSubwordTokenizer,
)


@dataclass(frozen=True)
class InstructionExample:
    category: str
    question: str
    answer: str


def load_instruction_examples(path: str) -> List[InstructionExample]:
    examples: List[InstructionExample] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path!r}.") from error
            required = {"category", "question", "answer"}
            if set(payload) != required:
                raise ValueError(
                    f"Line {line_number} of {path!r} must contain exactly "
                    f"{sorted(required)}."
                )
            values = {key: str(payload[key]).strip() for key in required}
            if not all(values.values()):
                raise ValueError(f"Line {line_number} of {path!r} contains an empty field.")
            examples.append(InstructionExample(**values))
    if not examples:
        raise ValueError(f"Instruction dataset {path!r} is empty.")
    return examples


def conversation_text(example: InstructionExample) -> str:
    return (
        f"{USER_TOKEN}\n{example.question}\n"
        f"{ASSISTANT_TOKEN}\n{example.answer}\n{END_TOKEN}"
    )


def encode_instruction_example(
    tokenizer: ByteSubwordTokenizer,
    example: InstructionExample,
) -> Tuple[List[int], List[int]]:
    user_id = tokenizer.special_token_id(USER_TOKEN)
    assistant_id = tokenizer.special_token_id(ASSISTANT_TOKEN)
    end_id = tokenizer.special_token_id(END_TOKEN)
    question_ids = tokenizer.encode(example.question)
    answer_ids = tokenizer.encode(example.answer)
    token_ids = [user_id, *question_ids, assistant_id, *answer_ids, end_id]

    # Labels are aligned with the inputs used to predict the next token.
    # Only answer tokens and the explicit end marker contribute to loss.
    input_ids = token_ids[:-1]
    labels = [-100] * len(input_ids)
    answer_start = 1 + len(question_ids) + 1
    for target_position in range(answer_start, len(token_ids)):
        labels[target_position - 1] = token_ids[target_position]
    return input_ids, labels


class InstructionDataset:
    """In-memory masked conversation examples with deterministic random batches."""

    def __init__(
        self,
        path: str,
        tokenizer: ByteSubwordTokenizer,
        context_length: int,
        seed: int = 1337,
    ) -> None:
        if context_length < 2:
            raise ValueError("context_length must be at least 2 for instruction tuning.")
        self.path = path
        self.context_length = context_length
        self.rng = np.random.default_rng(seed)
        self.examples = [
            encode_instruction_example(tokenizer, example)
            for example in load_instruction_examples(path)
        ]
        too_long = [
            len(input_ids)
            for input_ids, _ in self.examples
            if len(input_ids) > context_length
        ]
        if too_long:
            raise ValueError(
                f"Instruction example in {path!r} needs {max(too_long) + 1} tokens, "
                f"but context length is {context_length}. Shorten the example instead "
                "of silently dropping its answer."
            )
        self.supervised_tokens = sum(
            sum(label != -100 for label in labels) for _, labels in self.examples
        )
        if self.supervised_tokens < 1:
            raise ValueError(f"Instruction dataset {path!r} has no assistant labels.")

    def __len__(self) -> int:
        return len(self.examples)

    def _collate(
        self,
        indices: Sequence[int],
        device: torch.device,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_length = max(len(self.examples[index][0]) for index in indices)
        input_batch = np.full(
            (len(indices), sequence_length),
            pad_token_id,
            dtype=np.int64,
        )
        label_batch = np.full(
            (len(indices), sequence_length),
            -100,
            dtype=np.int64,
        )
        for row, index in enumerate(indices):
            input_ids, labels = self.examples[index]
            input_batch[row, : len(input_ids)] = input_ids
            label_batch[row, : len(labels)] = labels
        return (
            torch.from_numpy(input_batch).to(device=device),
            torch.from_numpy(label_batch).to(device=device),
        )

    def batch(
        self,
        batch_size: int,
        device: torch.device,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        indices = self.rng.integers(0, len(self.examples), size=batch_size)
        return self._collate(indices.tolist(), device, pad_token_id)

    def batches(
        self,
        batch_size: int,
        device: torch.device,
        pad_token_id: int,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        for start in range(0, len(self.examples), batch_size):
            indices = list(range(start, min(start + batch_size, len(self.examples))))
            yield self._collate(indices, device, pad_token_id)


def instruction_dataset_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
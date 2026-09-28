"""A compact learned byte-subword tokenizer.

The original project used one token per UTF-8 byte.  That is wonderfully easy
to understand, but it makes the model spend most of its context learning
spelling.  This tokenizer keeps the byte fallback (so every input is always
representable) and adds frequently occurring byte phrases learned from the
corpus.

It is intentionally dependency-free.  The saved JSON file is portable and
can be regenerated with ``prepare_dataset.py`` whenever the corpus changes.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Iterable, List, Optional


BYTE_VOCAB_SIZE = 256
EOS_TOKEN_ID = 256
PAD_TOKEN_ID = 257
SPECIAL_TOKEN_COUNT = 2


class ByteSubwordTokenizer:
    """Greedy longest-match byte tokenizer with a byte-level fallback."""

    def __init__(self, tokens: List[bytes]) -> None:
        if len(tokens) < BYTE_VOCAB_SIZE + SPECIAL_TOKEN_COUNT:
            raise ValueError("Tokenizer vocabulary is too small.")
        if tokens[:BYTE_VOCAB_SIZE] != [bytes([value]) for value in range(BYTE_VOCAB_SIZE)]:
            raise ValueError("Tokenizer must start with the 256 byte fallback tokens.")
        self.tokens = tokens
        self.vocab_size = len(tokens)
        self.eos_token_id = EOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        self._by_first_byte: List[List[tuple[bytes, int]]] = [[] for _ in range(256)]
        for token_id, token in enumerate(tokens[BYTE_VOCAB_SIZE + SPECIAL_TOKEN_COUNT :], start=BYTE_VOCAB_SIZE + SPECIAL_TOKEN_COUNT):
            self._by_first_byte[token[0]].append((token, token_id))
        for candidates in self._by_first_byte:
            candidates.sort(key=lambda item: len(item[0]), reverse=True)

    @classmethod
    def train(
        cls,
        sample: bytes,
        vocab_size: int = 4096,
        max_phrase_length: int = 12,
        min_frequency: int = 2,
    ) -> "ByteSubwordTokenizer":
        """Learn common byte phrases from a representative corpus sample.

        This is a frequency-based phrase vocabulary rather than a full BPE
        implementation.  It has the important properties needed here:
        deterministic training, lossless decoding, no external dependency,
        and much shorter sequences than raw bytes.
        """
        if vocab_size <= BYTE_VOCAB_SIZE + SPECIAL_TOKEN_COUNT:
            raise ValueError("vocab_size must leave room for learned phrases.")
        if max_phrase_length < 2:
            raise ValueError("max_phrase_length must be at least 2.")

        counts: Counter[bytes] = Counter()
        # Count within whitespace-delimited regions so a phrase never crosses
        # an arbitrary document boundary or combines unrelated words.
        regions = sample.split()
        for region in regions:
            limit = min(len(region), 256)
            region = region[:limit]
            for length in range(2, min(max_phrase_length, len(region)) + 1):
                for start in range(0, len(region) - length + 1):
                    counts[region[start : start + length]] += 1

        capacity = vocab_size - BYTE_VOCAB_SIZE - SPECIAL_TOKEN_COUNT
        ranked = sorted(
            (
                (phrase, count)
                for phrase, count in counts.items()
                if count >= min_frequency and len(phrase) >= 2
            ),
            key=lambda item: (item[1] * (len(item[0]) - 1), item[1], len(item[0]), item[0]),
            reverse=True,
        )
        phrases: List[bytes] = []
        seen = set()
        for phrase, _ in ranked:
            if phrase not in seen:
                phrases.append(phrase)
                seen.add(phrase)
            if len(phrases) >= capacity:
                break

        tokens = [bytes([value]) for value in range(BYTE_VOCAB_SIZE)]
        tokens.extend([b"", b""])  # EOS and PAD have no decoded bytes.
        tokens.extend(phrases)
        return cls(tokens)

    @classmethod
    def load(cls, path: str) -> "ByteSubwordTokenizer":
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        tokens = [bytes.fromhex(value) for value in payload["tokens"]]
        tokenizer = cls(tokens)
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError(f"Tokenizer metadata does not match {path!r}.")
        return tokenizer

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temporary_path = path + ".partial"
        payload = {
            "format": "byte-subword-v1",
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "tokens": [token.hex() for token in self.tokens],
        }
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, separators=(",", ":"))
        os.replace(temporary_path, path)

    def encode_bytes(self, data: bytes, add_eos: bool = False) -> List[int]:
        token_ids: List[int] = []
        position = 0
        while position < len(data):
            candidates = self._by_first_byte[data[position]]
            matched_id = int(data[position])
            matched_length = 1
            for phrase, token_id in candidates:
                if len(phrase) <= matched_length:
                    break
                if data.startswith(phrase, position):
                    matched_id = token_id
                    matched_length = len(phrase)
                    break
            token_ids.append(matched_id)
            position += matched_length
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        return self.encode_bytes(text.encode("utf-8"), add_eos=add_eos)

    def decode(self, token_ids: Iterable[int]) -> str:
        output = bytearray()
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in {self.eos_token_id, self.pad_token_id}:
                continue
            if token_id < 0 or token_id >= len(self.tokens):
                raise ValueError(f"Token ID {token_id} is outside the vocabulary.")
            output.extend(self.tokens[token_id])
        return bytes(output).decode("utf-8", errors="replace")


def load_tokenizer(path: Optional[str] = None) -> ByteSubwordTokenizer:
    """Load a trained tokenizer, with a useful legacy byte fallback."""
    path = path or "tokenizer.json"
    if os.path.exists(path):
        return ByteSubwordTokenizer.load(path)
    return ByteSubwordTokenizer([bytes([value]) for value in range(256)] + [b"", b""])


def encode(text: str) -> List[int]:
    """Compatibility helper; trained projects should use ``load_tokenizer``."""
    return load_tokenizer().encode(text)


def decode(token_ids: Iterable[int]) -> str:
    """Compatibility helper; trained projects should use ``load_tokenizer``."""
    return load_tokenizer().decode(token_ids)


VOCAB_SIZE = BYTE_VOCAB_SIZE
Tokenizer = ByteSubwordTokenizer
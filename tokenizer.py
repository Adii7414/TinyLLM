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
import hashlib
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional


BYTE_VOCAB_SIZE = 256
EOS_TOKEN_ID = 256
PAD_TOKEN_ID = 257
SPECIAL_TOKEN_COUNT = 2
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
INSTRUCTION_SPECIAL_TOKENS = (USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN)


class ByteSubwordTokenizer:
    """Greedy longest-match byte tokenizer with a byte-level fallback."""

    def __init__(
        self,
        tokens: List[bytes],
        special_tokens: Optional[Dict[str, int]] = None,
    ) -> None:
        if len(tokens) < BYTE_VOCAB_SIZE + SPECIAL_TOKEN_COUNT:
            raise ValueError("Tokenizer vocabulary is too small.")
        if tokens[:BYTE_VOCAB_SIZE] != [bytes([value]) for value in range(BYTE_VOCAB_SIZE)]:
            raise ValueError("Tokenizer must start with the 256 byte fallback tokens.")
        if tokens[EOS_TOKEN_ID] != b"" or tokens[PAD_TOKEN_ID] != b"":
            raise ValueError("EOS and PAD tokens must be empty-byte special tokens.")
        self.special_tokens = {
            "<|eos|>": EOS_TOKEN_ID,
            "<|pad|>": PAD_TOKEN_ID,
        }
        if special_tokens is not None:
            self.special_tokens.update(special_tokens)
        special_ids = set(self.special_tokens.values())
        if any(
            not isinstance(name, str) or not isinstance(token_id, int)
            for name, token_id in self.special_tokens.items()
        ):
            raise ValueError("Special-token names must map to integer token IDs.")
        if len(special_ids) != len(self.special_tokens):
            raise ValueError("Special-token IDs must be unique.")
        if any(token_id < 0 or token_id >= len(tokens) for token_id in special_ids):
            raise ValueError("Special-token ID is outside the tokenizer vocabulary.")
        if any(tokens[token_id] != b"" for token_id in special_ids):
            raise ValueError("Special-token entries must have empty byte payloads.")
        self.tokens = tokens
        self.vocab_size = len(tokens)
        self.eos_token_id = EOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        self._by_first_byte: List[List[tuple[bytes, int]]] = [[] for _ in range(256)]
        for token_id, token in enumerate(tokens):
            if token_id in special_ids:
                continue
            if not token:
                raise ValueError(
                    f"Non-special tokenizer entry {token_id} has an empty byte payload."
                )
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
        if payload.get("format") not in {"byte-subword-v1", "byte-subword-v2"}:
            raise ValueError(f"Unsupported tokenizer format in {path!r}.")
        if payload.get("eos_token_id") != EOS_TOKEN_ID:
            raise ValueError(f"Tokenizer EOS ID does not match {path!r}.")
        if payload.get("pad_token_id") != PAD_TOKEN_ID:
            raise ValueError(f"Tokenizer PAD ID does not match {path!r}.")
        if not isinstance(payload.get("tokens"), list):
            raise ValueError(f"Tokenizer token table is missing in {path!r}.")
        tokens = [bytes.fromhex(value) for value in payload["tokens"]]
        special_tokens = payload.get("special_tokens")
        if special_tokens is not None and not isinstance(special_tokens, dict):
            raise ValueError(f"Tokenizer special-token metadata is invalid in {path!r}.")
        tokenizer = cls(tokens, special_tokens=special_tokens)
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError(f"Tokenizer metadata does not match {path!r}.")
        return tokenizer

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temporary_path = path + ".partial"
        payload = {
            "format": "byte-subword-v2",
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "special_tokens": self.special_tokens,
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
        if not self.special_tokens:
            return self.encode_bytes(text.encode("utf-8"), add_eos=add_eos)
        markers = sorted(self.special_tokens, key=len, reverse=True)
        token_ids: List[int] = []
        position = 0
        while position < len(text):
            next_marker = min(
                (
                    (text.find(marker, position), marker)
                    for marker in markers
                    if text.find(marker, position) >= 0
                ),
                default=(len(text), ""),
            )
            marker_position, marker = next_marker
            if marker_position > position:
                token_ids.extend(self.encode_bytes(text[position:marker_position].encode("utf-8")))
            if marker:
                token_ids.append(self.special_tokens[marker])
                position = marker_position + len(marker)
            else:
                position = len(text)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        output = bytearray()
        special_ids = set(self.special_tokens.values())
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in special_ids:
                continue
            if token_id < 0 or token_id >= len(self.tokens):
                raise ValueError(f"Token ID {token_id} is outside the vocabulary.")
            output.extend(self.tokens[token_id])
        return bytes(output).decode("utf-8", errors="replace")

    def add_special_tokens(self, names: Iterable[str]) -> "ByteSubwordTokenizer":
        """Return a tokenizer with appended, non-byte control-token IDs."""
        special_tokens = dict(self.special_tokens)
        tokens = list(self.tokens)
        for name in names:
            if not name:
                raise ValueError("Special-token names must not be empty.")
            if name not in special_tokens:
                special_tokens[name] = len(tokens)
                tokens.append(b"")
        return ByteSubwordTokenizer(tokens, special_tokens=special_tokens)

    def special_token_id(self, name: str) -> int:
        try:
            return self.special_tokens[name]
        except KeyError as error:
            raise ValueError(f"Tokenizer does not define special token {name!r}.") from error

    def generation_end_token_id(self) -> int:
        """Return the response terminator when available, otherwise EOS."""
        return self.special_tokens.get(END_TOKEN, self.eos_token_id)


def load_tokenizer(path: str) -> ByteSubwordTokenizer:
    """Load the required trained tokenizer; never substitute a legacy tokenizer."""
    if not path:
        raise ValueError("A tokenizer path is required.")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tokenizer {path!r} does not exist. "
            "Run prepare_dataset.py before loading a model."
        )
    return ByteSubwordTokenizer.load(path)


def tokenizer_fingerprint(path: str) -> str:
    """Return the SHA-256 fingerprint of the exact tokenizer artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


Tokenizer = ByteSubwordTokenizer

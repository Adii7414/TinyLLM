"""A deliberately simple UTF-8 byte-level tokenizer.

The vocabulary is exactly the 256 possible byte values. This is not as
compact as BPE/subword tokenization, but it makes the mapping transparent:
every encoded token is one byte and every possible byte has an ID.
"""

from typing import Iterable, List, Union


VOCAB_SIZE = 256


def encode(text: str) -> List[int]:
    """Encode Unicode text as UTF-8 bytes represented by integer token IDs."""
    return list(text.encode("utf-8"))


def decode(token_ids: Iterable[int]) -> str:
    """Decode byte token IDs as UTF-8, replacing incomplete/invalid sequences."""
    values = [int(token) % VOCAB_SIZE for token in token_ids]
    return bytes(values).decode("utf-8", errors="replace")


class ByteTokenizer:
    """Small object wrapper useful for callers that prefer tokenizer methods."""

    vocab_size = VOCAB_SIZE

    def encode(self, text: str) -> List[int]:
        return encode(text)

    def decode(self, token_ids: Iterable[int]) -> str:
        return decode(token_ids)
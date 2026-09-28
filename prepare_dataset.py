"""Train a tokenizer and convert text into a disk-backed token stream."""

import argparse
import json
import os
from typing import Optional

import numpy as np

from config import DEFAULT_CONFIG
from tokenizer import ByteSubwordTokenizer


def prepare_dataset(
    source_path: str,
    output_path: str,
    metadata_path: str,
    tokenizer_path: str,
    vocab_size: int = DEFAULT_CONFIG.vocab_size,
    chunk_chars: int = 1_048_576,
    max_tokenizer_sample_bytes: int = 4_000_000,
) -> int:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temporary_path = output_path + ".partial"
    with open(source_path, "rb") as source:
        sample = source.read(max_tokenizer_sample_bytes)
    tokenizer = ByteSubwordTokenizer.train(sample, vocab_size=vocab_size)
    tokenizer.save(tokenizer_path)

    total_tokens = 0
    with open(source_path, "rb") as source, open(temporary_path, "wb") as output:
        while True:
            text_chunk = source.read(chunk_chars)
            if not text_chunk:
                break
            token_ids = tokenizer.encode_bytes(text_chunk)
            token_ids.append(tokenizer.eos_token_id)
            token_chunk = np.asarray(token_ids, dtype=np.uint16)
            output.write(token_chunk.tobytes())
            total_tokens += int(token_chunk.size)
    os.replace(temporary_path, output_path)
    metadata = {
        "source_path": source_path,
        "output_path": output_path,
        "token_count": total_tokens,
        "tokenizer_path": tokenizer_path,
        "dtype": "uint16",
        "vocab_size": tokenizer.vocab_size,
        "chunk_chars": chunk_chars,
        "max_tokenizer_sample_bytes": max_tokenizer_sample_bytes,
    }
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_CONFIG.dataset_source)
    parser.add_argument("--output", default=DEFAULT_CONFIG.dataset_path)
    parser.add_argument("--metadata", default=DEFAULT_CONFIG.dataset_meta_path)
    parser.add_argument("--tokenizer", default=DEFAULT_CONFIG.tokenizer_path)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_CONFIG.vocab_size)
    parser.add_argument("--chunk-chars", type=int, default=1_048_576)
    parser.add_argument("--max-tokenizer-sample-bytes", type=int, default=4_000_000)
    args = parser.parse_args()
    count = prepare_dataset(
        args.input,
        args.output,
        args.metadata,
        args.tokenizer,
        args.vocab_size,
        args.chunk_chars,
        args.max_tokenizer_sample_bytes,
    )
    print(f"Prepared {count:,} subword tokens from {args.input!r} -> {args.output!r}")
    print(f"Tokenizer vocabulary: {args.vocab_size:,} (saved to {args.tokenizer!r})")
    print(f"Metadata written to {args.metadata!r}")


if __name__ == "__main__":
    main()
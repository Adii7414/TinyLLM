"""Convert a text file into a disk-backed uint8 token stream in chunks."""

import argparse
import json
import os
from typing import Optional

import numpy as np

from config import DEFAULT_CONFIG


def prepare_dataset(
    source_path: str,
    output_path: str,
    metadata_path: str,
    chunk_chars: int = 1_048_576,
) -> int:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temporary_path = output_path + ".partial"
    total_tokens = 0
    with open(source_path, "r", encoding="utf-8") as source, open(temporary_path, "wb") as output:
        while True:
            text_chunk = source.read(chunk_chars)
            if not text_chunk:
                break
            token_chunk = np.frombuffer(text_chunk.encode("utf-8"), dtype=np.uint8)
            output.write(token_chunk.tobytes())
            total_tokens += int(token_chunk.size)
    os.replace(temporary_path, output_path)
    metadata = {
        "source_path": source_path,
        "output_path": output_path,
        "token_count": total_tokens,
        "dtype": "uint8",
        "vocab_size": 256,
        "chunk_chars": chunk_chars,
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
    parser.add_argument("--chunk-chars", type=int, default=1_048_576)
    args = parser.parse_args()
    count = prepare_dataset(args.input, args.output, args.metadata, args.chunk_chars)
    print(f"Prepared {count:,} byte tokens from {args.input!r} -> {args.output!r}")
    print(f"Metadata written to {args.metadata!r}")


if __name__ == "__main__":
    main()
"""Create deterministic document-level train/validation/test token files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from config import DEFAULT_CONFIG
from tokenizer import ByteSubwordTokenizer


MANIFEST_FORMAT = "document-split-v2"
TOKEN_DTYPE = "uint16"
DEFAULT_TRAIN_RATIO = 0.90
DEFAULT_VALIDATION_RATIO = 0.05
DEFAULT_TEST_RATIO = 0.05


def iter_documents(source_path: str) -> Iterator[bytes]:
    """Yield blank-line-delimited UTF-8 documents without arbitrary chunk EOS."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source corpus {source_path!r} does not exist.")
    if path.is_dir():
        for child in sorted(p for p in path.iterdir() if p.is_file()):
            yield from iter_documents(str(child))
        return

    document_lines: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as source:
        for line in source:
            if line.strip() == "":
                if document_lines:
                    document = "".join(document_lines).strip()
                    if document:
                        yield document.encode("utf-8")
                    document_lines = []
            else:
                document_lines.append(line)
    if document_lines:
        document = "".join(document_lines).strip()
        if document:
            yield document.encode("utf-8")


def document_split(
    document: bytes,
    split_seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> str:
    """Assign a document using a stable content-and-seed hash."""
    seed_bytes = str(split_seed).encode("ascii")
    digest = hashlib.sha256(seed_bytes + b"\0" + document).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: str, payload: Dict) -> None:
    temporary_path = path + ".partial"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, path)


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be positive.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")


def prepare_dataset(
    source_path: str,
    tokenizer_path: str,
    manifest_path: str,
    train_output_path: str,
    validation_output_path: str,
    test_output_path: str,
    vocab_size: int = DEFAULT_CONFIG.tokenizer_target_vocab_size,
    split_seed: int = DEFAULT_CONFIG.seed,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    max_tokenizer_sample_bytes: int = 4_000_000,
) -> Dict:
    """Split documents, train from train-only bytes, then tokenize each split."""
    validate_ratios(train_ratio, validation_ratio, test_ratio)
    if max_tokenizer_sample_bytes < 1:
        raise ValueError("max_tokenizer_sample_bytes must be positive.")
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocab_size must fit in the uint16 token file format.")

    split_paths = {
        "train": train_output_path,
        "validation": validation_output_path,
        "test": test_output_path,
    }
    for path in split_paths.values():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(tokenizer_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    # First pass: assign complete documents and collect a bounded tokenizer
    # sample from training documents only. No tokenization happens here.
    tokenizer_sample = bytearray()
    document_counts = {"train": 0, "validation": 0, "test": 0}
    document_bytes = {"train": 0, "validation": 0, "test": 0}
    source_digest = hashlib.sha256()
    for document in iter_documents(source_path):
        source_digest.update(len(document).to_bytes(8, "big"))
        source_digest.update(document)
        split = document_split(
            document, split_seed, train_ratio, validation_ratio, test_ratio
        )
        document_counts[split] += 1
        document_bytes[split] += len(document)
        if split == "train" and len(tokenizer_sample) < max_tokenizer_sample_bytes:
            remaining = max_tokenizer_sample_bytes - len(tokenizer_sample)
            tokenizer_sample.extend(document[:remaining])
            if len(tokenizer_sample) < max_tokenizer_sample_bytes:
                tokenizer_sample.extend(b"\n")

    if any(document_counts[split] == 0 for split in split_paths):
        raise ValueError(
            "The document split produced an empty partition. Add more documents "
            "or adjust the split ratios/seed."
        )

    tokenizer = ByteSubwordTokenizer.train(bytes(tokenizer_sample), vocab_size=vocab_size)
    tokenizer.save(tokenizer_path)

    temporary_paths = {name: path + ".partial" for name, path in split_paths.items()}
    token_counts = {name: 0 for name in split_paths}
    try:
        outputs = {
            name: open(path, "wb") for name, path in temporary_paths.items()
        }
        try:
            # Second pass: reassign the same documents and write one EOS per
            # document. No output depends on the size of an I/O read chunk.
            for document in iter_documents(source_path):
                split = document_split(
                    document, split_seed, train_ratio, validation_ratio, test_ratio
                )
                token_ids = tokenizer.encode_bytes(document, add_eos=True)
                np.asarray(token_ids, dtype=np.uint16).tofile(outputs[split])
                token_counts[split] += len(token_ids)
        finally:
            for output in outputs.values():
                output.close()
        for name, temporary_path in temporary_paths.items():
            os.replace(temporary_path, split_paths[name])
    except Exception:
        for temporary_path in temporary_paths.values():
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        raise

    tokenizer_hash = sha256_file(tokenizer_path)
    manifest = {
        "format": MANIFEST_FORMAT,
        "source": {
            "path": source_path,
            "document_count": sum(document_counts.values()),
            "document_bytes": sum(document_bytes.values()),
            "content_hash": source_digest.hexdigest(),
        },
        "split": {
            "seed": split_seed,
            "method": "sha256(seed + document_bytes)",
            "ratios": {
                "train": train_ratio,
                "validation": validation_ratio,
                "test": test_ratio,
            },
            "document_counts": document_counts,
            "document_bytes": document_bytes,
        },
        "tokenizer": {
            "path": tokenizer_path,
            "sha256": tokenizer_hash,
            "vocab_size": tokenizer.vocab_size,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "training_sample_bytes": len(tokenizer_sample),
            "training_source": "train_documents_only",
        },
        "preprocessing": {
            "document_format": "blank_line_delimited_utf8",
            "boundary_normalization": "strip_boundary_whitespace",
            "eos_policy": "one_eos_per_document",
            "dtype": TOKEN_DTYPE,
        },
        "splits": {
            name: {
                "path": path,
                "token_count": token_counts[name],
                "sha256": sha256_file(path),
            }
            for name, path in split_paths.items()
        },
    }
    atomic_json_write(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_CONFIG.dataset_source)
    parser.add_argument("--tokenizer", default=DEFAULT_CONFIG.tokenizer_path)
    parser.add_argument("--manifest", default=DEFAULT_CONFIG.dataset_manifest_path)
    parser.add_argument("--train-output", default="train_tokens.bin")
    parser.add_argument("--validation-output", default="validation_tokens.bin")
    parser.add_argument("--test-output", default="test_tokens.bin")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_CONFIG.tokenizer_target_vocab_size,
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--max-tokenizer-sample-bytes", type=int, default=4_000_000)
    args = parser.parse_args()
    manifest = prepare_dataset(
        source_path=args.input,
        tokenizer_path=args.tokenizer,
        manifest_path=args.manifest,
        train_output_path=args.train_output,
        validation_output_path=args.validation_output,
        test_output_path=args.test_output,
        vocab_size=args.vocab_size,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        max_tokenizer_sample_bytes=args.max_tokenizer_sample_bytes,
    )
    print(
        f"Prepared documents: train={manifest['split']['document_counts']['train']:,}, "
        f"validation={manifest['split']['document_counts']['validation']:,}, "
        f"test={manifest['split']['document_counts']['test']:,}"
    )
    for name, info in manifest["splits"].items():
        print(f"{name}: {info['token_count']:,} tokens -> {info['path']}")
    print(f"Tokenizer vocabulary: {manifest['tokenizer']['vocab_size']:,}")
    print(f"Manifest written to {args.manifest!r}")


if __name__ == "__main__":
    main()
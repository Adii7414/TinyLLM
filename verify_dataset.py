"""Verify every integrity and isolation contract of a prepared dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from prepare_dataset import (
    MANIFEST_FORMAT,
    TOKENIZER_MAX_PHRASE_LENGTH,
    TOKENIZER_MIN_FREQUENCY,
    TOKENIZER_TRAINING_ALGORITHM,
    document_split,
    iter_documents,
    sha256_file,
)
from tokenizer import ByteSubwordTokenizer


SPLITS = ("train", "validation", "test")
DTYPE_SIZES = {
    "uint8": np.dtype("uint8").itemsize,
    "uint16": np.dtype("uint16").itemsize,
    "int32": np.dtype("int32").itemsize,
}


class DatasetVerificationError(ValueError):
    """Raised when one or more dataset integrity contracts are violated."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(f"- {error}" for error in self.errors))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _same_hash(left: Any, right: Any) -> bool:
    return isinstance(left, str) and isinstance(right, str) and left == right


def _source_digest_update(digest: "hashlib._Hash", document: bytes) -> None:
    digest.update(len(document).to_bytes(8, "big"))
    digest.update(document)


def _manifest_value(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
    errors: List[str],
) -> Any:
    if key not in mapping:
        errors.append(f"Manifest is missing {location}.{key}.")
        return None
    return mapping[key]


def _resolve_recorded_path(path: Any, location: str, errors: List[str]) -> str | None:
    if not isinstance(path, str) or not path:
        errors.append(f"Manifest field {location} must be a non-empty path.")
        return None
    return path


def _verify_token_file(
    name: str,
    info: Mapping[str, Any],
    dtype_name: str,
    vocab_size: int,
    errors: List[str],
) -> np.memmap | None:
    path = _resolve_recorded_path(info.get("path"), f"splits.{name}", errors)
    token_count = info.get("token_count")
    if not _is_int(token_count) or token_count < 0:
        errors.append(f"Manifest splits.{name}.token_count must be a non-negative integer.")
        token_count = None

    if path is None or not os.path.isfile(path):
        if path is not None:
            errors.append(f"{name} token file {path!r} does not exist.")
        return None
    if dtype_name not in DTYPE_SIZES:
        return None

    expected_size = token_count * DTYPE_SIZES[dtype_name] if token_count is not None else None
    actual_size = os.path.getsize(path)
    if expected_size is not None and actual_size != expected_size:
        errors.append(
            f"{name} token file size is {actual_size} bytes; metadata requires "
            f"{expected_size} bytes for {token_count} {dtype_name} tokens."
        )
    try:
        actual_hash = sha256_file(path)
        if not _same_hash(info.get("sha256"), actual_hash):
            errors.append(f"{name} token file SHA-256 does not match the manifest.")
    except OSError as error:
        errors.append(f"Could not hash {name} token file {path!r}: {error}")

    try:
        tokens = np.memmap(path, dtype=np.dtype(dtype_name), mode="r")
    except (OSError, ValueError) as error:
        errors.append(f"Could not read {name} token file {path!r}: {error}")
        return None

    if token_count is not None and tokens.size != token_count:
        errors.append(
            f"{name} token file contains {tokens.size} tokens; metadata says {token_count}."
        )
    if tokens.size:
        minimum = int(tokens.min())
        maximum = int(tokens.max())
        if minimum < 0:
            errors.append(f"{name} token file contains negative token ID {minimum}.")
        if maximum >= vocab_size:
            errors.append(
                f"{name} token file contains token ID {maximum}, but tokenizer "
                f"vocabulary size is {vocab_size}."
            )
    return tokens


def verify_dataset(manifest_path: str = "dataset_manifest.json") -> Dict[str, Any]:
    """Verify a prepared dataset, raising one error containing all failures."""
    errors: List[str] = []
    manifest: Dict[str, Any] = {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
        if not isinstance(loaded, dict):
            raise ValueError("the top-level JSON value is not an object")
        manifest = loaded
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DatasetVerificationError(
            [f"Could not load dataset manifest {manifest_path!r}: {error}"]
        ) from error

    if manifest.get("format") != MANIFEST_FORMAT:
        errors.append(
            f"Manifest format {manifest.get('format')!r} is not the current "
            f"{MANIFEST_FORMAT!r}; the legacy single-file dataset is not supported."
        )

    source = manifest.get("source")
    split = manifest.get("split")
    tokenizer_metadata = manifest.get("tokenizer")
    preprocessing = manifest.get("preprocessing")
    splits = manifest.get("splits")
    for value, name in (
        (source, "source"),
        (split, "split"),
        (tokenizer_metadata, "tokenizer"),
        (preprocessing, "preprocessing"),
        (splits, "splits"),
    ):
        if not isinstance(value, dict):
            errors.append(f"Manifest field {name} must be an object.")
    if not all(isinstance(value, dict) for value in (source, split, tokenizer_metadata, preprocessing, splits)):
        raise DatasetVerificationError(errors)

    source_path = _resolve_recorded_path(source.get("path"), "source", errors)
    split_seed = split.get("seed")
    ratios = split.get("ratios")
    if not _is_int(split_seed):
        errors.append("Manifest split.seed must be an integer.")
        split_seed = 0
    if not isinstance(ratios, dict):
        errors.append("Manifest split.ratios must be an object.")
        ratios = {}
    ratio_values = [ratios.get(name) for name in SPLITS]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in ratio_values):
        errors.append("Manifest split.ratios must contain numeric train, validation, and test values.")
        ratio_values = [0.0, 0.0, 0.0]
    train_ratio, validation_ratio, test_ratio = (float(value) for value in ratio_values)
    if any(value <= 0 for value in (train_ratio, validation_ratio, test_ratio)):
        errors.append("Manifest split ratios must all be positive.")
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        errors.append("Manifest split ratios must sum to 1.")

    dtype_name = preprocessing.get("dtype")
    if dtype_name not in DTYPE_SIZES:
        errors.append(
            f"Manifest preprocessing.dtype {dtype_name!r} is unsupported; "
            f"expected one of {sorted(DTYPE_SIZES)}."
        )
        dtype_name = "uint16"

    tokenizer_path = _resolve_recorded_path(tokenizer_metadata.get("path"), "tokenizer", errors)
    tokenizer: ByteSubwordTokenizer | None = None
    if tokenizer_path is None or not os.path.isfile(tokenizer_path):
        if tokenizer_path is not None:
            errors.append(f"Tokenizer file {tokenizer_path!r} does not exist.")
    else:
        try:
            tokenizer = ByteSubwordTokenizer.load(tokenizer_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"Could not load tokenizer {tokenizer_path!r}: {error}")

    vocab_size = tokenizer.vocab_size if tokenizer is not None else 0
    metadata_vocab_size = tokenizer_metadata.get("vocab_size")
    if not _is_int(metadata_vocab_size) or metadata_vocab_size < 1:
        errors.append("Manifest tokenizer.vocab_size must be a positive integer.")
        metadata_vocab_size = vocab_size
    elif tokenizer is not None and tokenizer.vocab_size != metadata_vocab_size:
        errors.append(
            f"Tokenizer vocabulary has {tokenizer.vocab_size} entries; "
            f"manifest says {metadata_vocab_size}."
        )
    vocab_size = int(metadata_vocab_size)

    if tokenizer is not None:
        for key, actual in (
            ("eos_token_id", tokenizer.eos_token_id),
            ("pad_token_id", tokenizer.pad_token_id),
        ):
            if tokenizer_metadata.get(key) != actual:
                errors.append(
                    f"Manifest tokenizer.{key}={tokenizer_metadata.get(key)!r} "
                    f"does not match the tokenizer ({actual})."
                )
        try:
            actual_hash = sha256_file(tokenizer_path)
            if not _same_hash(tokenizer_metadata.get("sha256"), actual_hash):
                errors.append("Tokenizer SHA-256 does not match the manifest.")
        except OSError as error:
            errors.append(f"Could not hash tokenizer {tokenizer_path!r}: {error}")

    if tokenizer_metadata.get("training_source") != "train_documents_only":
        errors.append("Tokenizer provenance is not marked as train_documents_only.")
    if tokenizer_metadata.get("training_algorithm") != TOKENIZER_TRAINING_ALGORITHM:
        errors.append("Tokenizer training algorithm metadata is missing or unsupported.")
    if tokenizer_metadata.get("max_phrase_length") != TOKENIZER_MAX_PHRASE_LENGTH:
        errors.append("Tokenizer max_phrase_length metadata does not match the pipeline.")
    if tokenizer_metadata.get("min_frequency") != TOKENIZER_MIN_FREQUENCY:
        errors.append("Tokenizer min_frequency metadata does not match the pipeline.")

    for name in SPLITS:
        if name not in splits or not isinstance(splits[name], dict):
            errors.append(f"Manifest is missing the {name!r} split metadata.")
    if any(name not in splits or not isinstance(splits[name], dict) for name in SPLITS):
        raise DatasetVerificationError(errors)

    token_files: Dict[str, np.memmap | None] = {}
    for name in SPLITS:
        token_files[name] = _verify_token_file(
            name, splits[name], dtype_name, vocab_size, errors
        )

    if source_path is None or not os.path.exists(source_path):
        if source_path is not None:
            errors.append(f"Source corpus {source_path!r} does not exist.")
        raise DatasetVerificationError(errors)

    document_counts = {name: 0 for name in SPLITS}
    document_bytes = {name: 0 for name in SPLITS}
    document_hashes = {name: set() for name in SPLITS}
    documents_by_split: Dict[str, List[bytes]] = {name: [] for name in SPLITS}
    tokenizer_sample = bytearray()
    sample_limit = tokenizer_metadata.get("training_sample_limit_bytes")
    if not _is_int(sample_limit) or sample_limit < 1:
        errors.append("Manifest tokenizer.training_sample_limit_bytes must be positive.")
        sample_limit = 0
    source_digest = hashlib.sha256()
    try:
        for document in iter_documents(source_path):
            _source_digest_update(source_digest, document)
            assigned = document_split(
                document,
                split_seed,
                train_ratio,
                validation_ratio,
                test_ratio,
            )
            document_counts[assigned] += 1
            document_bytes[assigned] += len(document)
            documents_by_split[assigned].append(document)
            document_hash = hashlib.sha256(document).hexdigest()
            document_hashes[assigned].add(document_hash)
            if assigned == "train" and len(tokenizer_sample) < sample_limit:
                remaining = sample_limit - len(tokenizer_sample)
                tokenizer_sample.extend(document[:remaining])
                if len(tokenizer_sample) < sample_limit:
                    tokenizer_sample.extend(b"\n")
    except (OSError, UnicodeError, ValueError) as error:
        errors.append(f"Could not iterate source corpus {source_path!r}: {error}")

    expected_source_hash = source.get("content_hash")
    actual_source_hash = source_digest.hexdigest()
    if not _same_hash(expected_source_hash, actual_source_hash):
        errors.append("Source corpus content hash does not match the manifest.")
    if source.get("document_count") != sum(document_counts.values()):
        errors.append(
            f"Source document count is {sum(document_counts.values())}; "
            f"manifest says {source.get('document_count')}."
        )
    if source.get("document_bytes") != sum(document_bytes.values()):
        errors.append(
            f"Source document bytes are {sum(document_bytes.values())}; "
            f"manifest says {source.get('document_bytes')}."
        )

    for name in SPLITS:
        if split.get("document_counts", {}).get(name) != document_counts[name]:
            errors.append(
                f"{name} document count is {document_counts[name]}; "
                f"manifest says {split.get('document_counts', {}).get(name)}."
            )
        if split.get("document_bytes", {}).get(name) != document_bytes[name]:
            errors.append(
                f"{name} document bytes are {document_bytes[name]}; "
                f"manifest says {split.get('document_bytes', {}).get(name)}."
            )

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = document_hashes[left] & document_hashes[right]
            if overlap:
                errors.append(
                    f"{left} and {right} contain {len(overlap)} overlapping document(s)."
                )

    if tokenizer is not None:
        actual_sample_hash = hashlib.sha256(tokenizer_sample).hexdigest()
        if tokenizer_metadata.get("training_sample_bytes") != len(tokenizer_sample):
            errors.append(
                f"Reconstructed train-only tokenizer sample has {len(tokenizer_sample)} "
                f"bytes; manifest says {tokenizer_metadata.get('training_sample_bytes')}."
            )
        if not _same_hash(tokenizer_metadata.get("training_sample_sha256"), actual_sample_hash):
            errors.append("Reconstructed train-only tokenizer sample hash does not match the manifest.")
        try:
            expected_tokenizer = ByteSubwordTokenizer.train(
                bytes(tokenizer_sample),
                vocab_size=vocab_size,
                max_phrase_length=TOKENIZER_MAX_PHRASE_LENGTH,
                min_frequency=TOKENIZER_MIN_FREQUENCY,
            )
            if expected_tokenizer.tokens != tokenizer.tokens:
                errors.append(
                    "Tokenizer does not match a tokenizer trained from the manifest's "
                    "TRAIN documents only."
                )
            if expected_tokenizer.special_tokens != tokenizer.special_tokens:
                errors.append("Tokenizer special-token table does not match train-only regeneration.")
        except (ValueError, TypeError) as error:
            errors.append(f"Could not regenerate the train-only tokenizer: {error}")

    # Re-encode every source document and compare it to the corresponding split.
    # This proves token counts, token content, EOS placement, and split isolation
    # together instead of trusting only aggregate metadata.
    if tokenizer is not None and all(token_files[name] is not None for name in SPLITS):
        for name in SPLITS:
            tokens = token_files[name]
            assert tokens is not None
            offset = 0
            eos_positions: List[int] = []
            for document_index, document in enumerate(documents_by_split[name]):
                expected = np.asarray(
                    tokenizer.encode_bytes(document, add_eos=True), dtype=np.dtype(dtype_name)
                )
                end = offset + expected.size
                actual = tokens[offset:end]
                if actual.size != expected.size or not np.array_equal(actual, expected):
                    errors.append(
                        f"{name} token stream does not match source document "
                        f"{document_index} at token offset {offset}."
                    )
                    break
                if expected[-1] != tokenizer.eos_token_id or np.any(
                    expected[:-1] == tokenizer.eos_token_id
                ):
                    errors.append(
                        f"{name} document {document_index} violates the one-EOS-at-boundary policy."
                    )
                eos_positions.append(end - 1)
                offset = end
            if offset != tokens.size:
                errors.append(
                    f"{name} token stream has {tokens.size} tokens, but source documents "
                    f"re-encode to {offset} tokens."
                )
            actual_eos_positions = np.flatnonzero(tokens == tokenizer.eos_token_id).tolist()
            if actual_eos_positions != eos_positions:
                errors.append(
                    f"{name} EOS positions do not match the ends of its source documents."
                )

    if errors:
        raise DatasetVerificationError(errors)

    return {
        "manifest": manifest_path,
        "source": source_path,
        "splits": {
            name: {
                "documents": document_counts[name],
                "tokens": int(splits[name]["token_count"]),
            }
            for name in SPLITS
        },
        "tokenizer_vocab_size": vocab_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset_manifest.json")
    args = parser.parse_args()
    try:
        result = verify_dataset(args.manifest)
    except DatasetVerificationError as error:
        print("DATASET VERIFICATION FAILED", file=sys.stderr)
        print(error, file=sys.stderr)
        raise SystemExit(1)
    print(f"Dataset verification passed: {result['manifest']}")
    print(f"Source corpus: {result['source']}")
    for name, info in result["splits"].items():
        print(f"{name}: {info['documents']:,} documents, {info['tokens']:,} tokens")
    print(f"Tokenizer vocabulary: {result['tokenizer_vocab_size']:,}")


if __name__ == "__main__":
    main()
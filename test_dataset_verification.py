"""Tests for dataset preparation and strict artifact verification."""

import json
import tempfile
import unittest
from pathlib import Path

from prepare_dataset import prepare_dataset
from verify_dataset import DatasetVerificationError, verify_dataset


class DatasetVerificationTests(unittest.TestCase):
    def test_preparation_and_verification_cover_all_dataset_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text(
                "\n\n".join(f"Document {index}: aircraft data." for index in range(60)),
                encoding="utf-8",
            )
            manifest_path = root / "dataset_manifest.json"
            prepare_dataset(
                source_path=str(source),
                tokenizer_path=str(root / "tokenizer.json"),
                manifest_path=str(manifest_path),
                train_output_path=str(root / "train_tokens.bin"),
                validation_output_path=str(root / "validation_tokens.bin"),
                test_output_path=str(root / "test_tokens.bin"),
                vocab_size=300,
                split_seed=1337,
                train_ratio=0.8,
                validation_ratio=0.1,
                test_ratio=0.1,
                max_tokenizer_sample_bytes=1000,
            )
            result = verify_dataset(str(manifest_path))
            self.assertEqual(result["splits"]["train"]["documents"] +
                             result["splits"]["validation"]["documents"] +
                             result["splits"]["test"]["documents"], 60)

    def test_verification_rejects_tampered_token_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("\n\n".join(f"Doc {index}" for index in range(30)), encoding="utf-8")
            manifest_path = root / "dataset_manifest.json"
            prepare_dataset(
                source_path=str(source),
                tokenizer_path=str(root / "tokenizer.json"),
                manifest_path=str(manifest_path),
                train_output_path=str(root / "train_tokens.bin"),
                validation_output_path=str(root / "validation_tokens.bin"),
                test_output_path=str(root / "test_tokens.bin"),
                vocab_size=300,
                train_ratio=0.8,
                validation_ratio=0.1,
                test_ratio=0.1,
                max_tokenizer_sample_bytes=1000,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_path = Path(manifest["splits"]["train"]["path"])
            tampered = bytearray(train_path.read_bytes())
            tampered[0] ^= 1
            train_path.write_bytes(tampered)
            with self.assertRaisesRegex(DatasetVerificationError, "SHA-256"):
                verify_dataset(str(manifest_path))


if __name__ == "__main__":
    unittest.main()
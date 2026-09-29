"""Exact tokenizer contract and round-trip tests."""

import tempfile
import unittest
from pathlib import Path

import tokenizer as tokenizer_module
from tokenizer import (
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    ByteSubwordTokenizer,
    load_tokenizer,
    tokenizer_fingerprint,
)


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = ByteSubwordTokenizer.train(
            (
                "A321neo systems and navigation. "
                "Unicode नमस्ते мир. "
                "Repeated aviation words make useful phrases. "
            ).encode("utf-8"),
            vocab_size=300,
        )

    def test_exact_unicode_round_trip(self) -> None:
        text = "Aircraft: A321neo — नमस्ते мир — café."
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_exact_byte_round_trip(self) -> None:
        data = bytes(range(256))
        token_ids = self.tokenizer.encode_bytes(data)
        self.assertEqual(b"".join(self.tokenizer.tokens[i] for i in token_ids), data)

    def test_special_tokens_are_explicit_and_lossless(self) -> None:
        text = "document boundary"
        token_ids = self.tokenizer.encode(text, add_eos=True)
        self.assertEqual(token_ids[-1], EOS_TOKEN_ID)
        self.assertEqual(self.tokenizer.decode(token_ids), text)
        self.assertEqual(self.tokenizer.pad_token_id, PAD_TOKEN_ID)

    def test_saved_tokenizer_preserves_contract_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "tokenizer.json")
            self.tokenizer.save(path)
            loaded = load_tokenizer(path)
            self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(loaded.eos_token_id, EOS_TOKEN_ID)
            self.assertEqual(loaded.pad_token_id, PAD_TOKEN_ID)
            self.assertEqual(
                loaded.decode(loaded.encode("round trip")),
                "round trip",
            )
            self.assertEqual(len(tokenizer_fingerprint(path)), 64)

    def test_missing_tokenizer_never_falls_back(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_tokenizer("/definitely/missing/tokenizer.json")

    def test_stale_vocab_constant_is_removed(self) -> None:
        self.assertFalse(hasattr(tokenizer_module, "VOCAB_SIZE"))


if __name__ == "__main__":
    unittest.main()
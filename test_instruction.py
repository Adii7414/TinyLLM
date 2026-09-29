"""Instruction-data formatting, masking, and vocabulary expansion tests."""

import tempfile
import unittest
from pathlib import Path

import torch

from config import Config
from finetune import load_instruction_model
from instruction_dataset import (
    InstructionDataset,
    InstructionExample,
    conversation_text,
    encode_instruction_example,
)
from model import GPTModel
from tokenizer import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    USER_TOKEN,
    ByteSubwordTokenizer,
)


class InstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_tokenizer = ByteSubwordTokenizer.train(
            b"aviation flight plan guidance aircraft system " * 10,
            vocab_size=300,
        )
        cls.tokenizer = cls.base_tokenizer.add_special_tokens(
            (USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN)
        )

    def test_roles_are_real_tokens_and_round_trip_as_control_ids(self) -> None:
        example = InstructionExample(
            "factual",
            "What is the FCU?",
            "It is the short-term guidance control panel.",
        )
        encoded = self.tokenizer.encode(conversation_text(example))
        self.assertEqual(encoded[0], self.tokenizer.special_token_id(USER_TOKEN))
        self.assertIn(self.tokenizer.special_token_id(ASSISTANT_TOKEN), encoded)
        self.assertEqual(encoded[-1], self.tokenizer.special_token_id(END_TOKEN))
        self.assertEqual(self.tokenizer.vocab_size, self.base_tokenizer.vocab_size + 3)

    def test_only_assistant_response_and_end_are_supervised(self) -> None:
        example = InstructionExample(
            "why_how",
            "Why check the FMA?",
            "It confirms the active guidance mode.",
        )
        input_ids, labels = encode_instruction_example(self.tokenizer, example)
        assistant_id = self.tokenizer.special_token_id(ASSISTANT_TOKEN)
        assistant_position = input_ids.index(assistant_id)
        self.assertTrue(all(label == -100 for label in labels[:assistant_position]))
        self.assertTrue(any(label != -100 for label in labels[assistant_position:]))
        self.assertEqual(labels[-1], self.tokenizer.special_token_id(END_TOKEN))

    def test_instruction_dataset_batches_have_masked_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                '{"category":"factual","question":"What is FCU?",'
                '"answer":"A guidance control panel."}\n',
                encoding="utf-8",
            )
            dataset = InstructionDataset(str(path), self.tokenizer, 64)
            inputs, labels = dataset.batch(2, torch.device("cpu"), self.tokenizer.pad_token_id)
            self.assertEqual(inputs.shape, labels.shape)
            self.assertTrue(torch.all(labels[labels == -100] == -100))
            self.assertGreater(dataset.supervised_tokens, 0)

    def test_pretrained_weights_are_preserved_when_vocab_expands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            base_path = directory_path / "base_tokenizer.json"
            instruction_path = directory_path / "instruction_tokenizer.json"
            self.base_tokenizer.save(str(base_path))
            self.tokenizer.save(str(instruction_path))
            config = Config(
                tokenizer_path=str(base_path),
                vocab_size=self.base_tokenizer.vocab_size,
                embedding_dim=12,
                num_layers=1,
                num_heads=3,
                feed_forward_dim=24,
                context_length=32,
                dropout=0.0,
            )
            torch.manual_seed(11)
            base_model = GPTModel(config)
            checkpoint = {"config": config.to_dict(), "model_state": base_model.state_dict()}
            expanded, _ = load_instruction_model(
                checkpoint,
                str(instruction_path),
                torch.device("cpu"),
            )
            self.assertEqual(expanded.config.vocab_size, self.tokenizer.vocab_size)
            self.assertTrue(
                torch.equal(
                    expanded.token_embedding.weight[: config.vocab_size],
                    base_model.token_embedding.weight,
                )
            )


if __name__ == "__main__":
    unittest.main()
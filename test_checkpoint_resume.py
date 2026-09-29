"""Checkpoint completeness and deterministic resume tests."""

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import Config
from dataset import TokenDataset
from model import GPTModel
from train import (
    LearningRateScheduler,
    amp_dtype_name,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)


def assert_nested_equal(test_case: unittest.TestCase, left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        test_case.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        test_case.assertTrue(np.array_equal(left, right))
    elif isinstance(left, dict):
        test_case.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_equal(test_case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test_case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            assert_nested_equal(test_case, left_item, right_item)
    else:
        test_case.assertEqual(left, right)


class CheckpointResumeTests(unittest.TestCase):
    def make_config(self, directory: str, training_steps: int) -> Config:
        return Config(
            dataset_manifest_path=str(Path(directory) / "dataset_manifest.json"),
            tokenizer_path=str(Path(directory) / "tokenizer.json"),
            dataset_identity={
                "manifest_sha256": "manifest-hash",
                "source_content_hash": "source-hash",
                "split_hashes": {
                    "train": "train-hash",
                    "validation": "validation-hash",
                    "test": "test-hash",
                },
                "format": "document-split-v2",
            },
            vocab_size=32,
            context_length=8,
            embedding_dim=16,
            num_layers=1,
            num_heads=1,
            feed_forward_dim=32,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=1,
            training_steps=training_steps,
            eval_interval=1,
            eval_steps=1,
            checkpoint_interval=1,
            amp_dtype="none",
        )

    def make_runtime(self, directory: str, config: Config):
        data_paths = []
        for name, seed in (("train", 11), ("validation", 22), ("test", 33)):
            path = Path(directory) / f"{name}.bin"
            (np.arange(seed, seed + 256, dtype=np.uint16) % config.vocab_size).tofile(path)
            data_paths.append(str(path))
        datasets = [
            TokenDataset(path, config.context_length, seed, config.dataset_dtype)
            for path, seed in zip(data_paths, (1, 2, 3))
        ]
        model = GPTModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        scheduler = LearningRateScheduler(optimizer, config)
        return model, optimizer, scaler, scheduler, datasets

    @staticmethod
    def train_one_step(model, optimizer, scheduler, dataset, step):
        x, y = dataset.batch(2, torch.device("cpu"))
        _, loss = model(x, y)
        loss.backward()
        learning_rate = scheduler.step(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach()), learning_rate

    def test_resume_restores_complete_state_and_allows_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(directory, training_steps=2)
            seed_everything(1234)
            model, optimizer, scaler, scheduler, datasets = self.make_runtime(
                directory, config
            )
            for step in (1, 2):
                self.train_one_step(model, optimizer, scheduler, datasets[0], step)

            checkpoint_path = str(Path(directory) / "checkpoint.pt")
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scaler,
                scheduler,
                *datasets,
                config,
                2,
                1.25,
                1.5,
                1.5,
            )
            self.assertTrue(Path(checkpoint_path).exists())
            self.assertFalse(Path(checkpoint_path + ".partial").exists())
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["checkpoint_format"], "training-checkpoint-v2")
            for key in (
                "model_state",
                "optimizer_state",
                "scaler_state",
                "scheduler_state",
                "step",
                "best_val_loss",
                "torch_rng_state",
                "numpy_rng_state",
                "python_rng_state",
                "train_dataset_rng_state",
                "validation_dataset_rng_state",
                "test_dataset_rng_state",
            ):
                self.assertIn(key, saved)

            expected_model = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            expected_optimizer = optimizer.state_dict()
            expected_scheduler = scheduler.state_dict()
            expected_global_torch = torch.get_rng_state().clone()
            expected_global_numpy = np.random.get_state()
            expected_global_python = random.getstate()
            expected_batches = [dataset.batch(2, torch.device("cpu")) for dataset in datasets]

            extended_config = replace(config, training_steps=4)
            seed_everything(9999)
            resumed_model, resumed_optimizer, resumed_scaler, resumed_scheduler, resumed_datasets = (
                self.make_runtime(directory, extended_config)
            )
            step, best_loss, _ = load_checkpoint(
                checkpoint_path,
                resumed_model,
                resumed_optimizer,
                resumed_scaler,
                resumed_scheduler,
                *resumed_datasets,
                torch.device("cpu"),
                extended_config,
            )

            self.assertEqual(step, 2)
            self.assertEqual(best_loss, 1.5)
            self.assertEqual(resumed_scheduler.step_num, 2)
            self.assertEqual(resumed_scheduler.schedule_config["total_steps"], 2)
            assert_nested_equal(self, expected_model, resumed_model.state_dict())
            assert_nested_equal(self, expected_optimizer, resumed_optimizer.state_dict())
            assert_nested_equal(self, expected_scheduler, resumed_scheduler.state_dict())
            self.assertTrue(torch.equal(expected_global_torch, torch.get_rng_state()))
            assert_nested_equal(self, expected_global_numpy, np.random.get_state())
            assert_nested_equal(self, expected_global_python, random.getstate())
            for expected, resumed in zip(
                expected_batches,
                [dataset.batch(2, torch.device("cpu")) for dataset in resumed_datasets],
            ):
                self.assertTrue(torch.equal(expected[0], resumed[0]))
                self.assertTrue(torch.equal(expected[1], resumed[1]))

            _, learning_rate = self.train_one_step(
                resumed_model, resumed_optimizer, resumed_scheduler, resumed_datasets[0], 3
            )
            self.assertEqual(resumed_scheduler.step_num, 3)
            self.assertEqual(learning_rate, config.min_learning_rate)

    def test_incompatible_resume_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(directory, training_steps=2)
            seed_everything(1234)
            model, optimizer, scaler, scheduler, datasets = self.make_runtime(
                directory, config
            )
            self.train_one_step(model, optimizer, scheduler, datasets[0], 1)
            checkpoint_path = str(Path(directory) / "checkpoint.pt")
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scaler,
                scheduler,
                *datasets,
                config,
                1,
                1.0,
                1.0,
                1.0,
            )
            incompatible = replace(config, training_steps=2, context_length=7)
            runtime = self.make_runtime(directory, incompatible)
            with self.assertRaisesRegex(ValueError, "context_length"):
                load_checkpoint(
                    checkpoint_path,
                    runtime[0],
                    runtime[1],
                    runtime[2],
                    runtime[3],
                    *runtime[4],
                    torch.device("cpu"),
                    incompatible,
                )


if __name__ == "__main__":
    unittest.main()
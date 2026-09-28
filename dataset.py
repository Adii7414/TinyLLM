"""Disk-backed random contiguous batches for next-token prediction."""

import os
from typing import Optional, Tuple

import numpy as np
import torch


class TokenDataset:
    """Read uint8 token data through a NumPy memory map.

    The complete token stream stays on disk. Only the small sampled batches are
    copied into PyTorch tensors.
    """

    def __init__(
        self,
        path: str,
        context_length: int,
        validation_split: float = 0.1,
        mode: str = "train",
        seed: int = 1337,
        dtype: str = "uint16",
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Token file {path!r} does not exist. Run prepare_dataset.py first."
            )
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")
        self.path = path
        self.context_length = context_length
        if dtype not in {"uint8", "uint16", "int32"}:
            raise ValueError("dtype must be one of: uint8, uint16, int32")
        self.tokens = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.total_tokens = int(self.tokens.shape[0])
        if self.total_tokens <= context_length + 1:
            raise ValueError(
                f"Dataset contains {self.total_tokens} tokens, but context length "
                f"is {context_length}; add more training text."
            )
        split = int(self.total_tokens * (1.0 - validation_split))
        # Both partitions need one extra token for the shifted target.
        split = max(context_length + 2, min(split, self.total_tokens - context_length - 2))
        if mode == "train":
            self.start, self.end = 0, split
        else:
            self.start, self.end = split, self.total_tokens
        if self.end - self.start <= context_length + 1:
            raise ValueError("The validation split is too small for this context length.")
        self.mode = mode
        self.rng = np.random.default_rng(seed + (0 if mode == "train" else 1))

    def __len__(self) -> int:
        return self.end - self.start - self.context_length

    def batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = self.end - self.context_length - 1
        offsets = self.rng.integers(self.start, max_start + 1, size=batch_size)
        x = np.stack([self.tokens[i : i + self.context_length] for i in offsets]).astype(
            np.int64, copy=False
        )
        y = np.stack(
            [self.tokens[i + 1 : i + self.context_length + 1] for i in offsets]
        ).astype(np.int64, copy=False)
        return (
            torch.from_numpy(x).to(device=device, non_blocking=True),
            torch.from_numpy(y).to(device=device, non_blocking=True),
        )


def get_batch(
    dataset: TokenDataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional convenience wrapper used by training and benchmarking."""
    return dataset.batch(batch_size, device)
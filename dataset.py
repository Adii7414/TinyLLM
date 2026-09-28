"""Disk-backed random contiguous batches for one pre-split token file."""

import os
from typing import Tuple

import numpy as np
import torch


class TokenDataset:
    """Read a single train/validation/test token stream through memmap.

    Splitting is deliberately not performed here.  The preprocessing stage
    creates separate files at document boundaries so the loader cannot
    accidentally reintroduce the old token-offset split.
    """

    def __init__(
        self,
        path: str,
        context_length: int,
        seed: int = 1337,
        dtype: str = "uint16",
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Token file {path!r} does not exist. Run prepare_dataset.py first."
            )
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if dtype not in {"uint8", "uint16", "int32"}:
            raise ValueError("dtype must be one of: uint8, uint16, int32")
        self.path = path
        self.context_length = context_length
        self.dtype = dtype
        self.tokens = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.total_tokens = int(self.tokens.shape[0])
        if self.total_tokens <= context_length + 1:
            raise ValueError(
                f"Dataset {path!r} contains {self.total_tokens} tokens, but context "
                f"length is {context_length}; add more training text."
            )
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.total_tokens - self.context_length

    def batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        max_start = self.total_tokens - self.context_length - 1
        offsets = self.rng.integers(0, max_start + 1, size=batch_size)
        x = np.stack(
            [self.tokens[i : i + self.context_length] for i in offsets]
        ).astype(np.int64, copy=False)
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
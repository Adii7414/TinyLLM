"""Central configuration for the educational byte-level GPT project."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Config:
    # Data
    dataset_path: str = "training_tokens.bin"
    dataset_source: str = "training_data.txt"
    dataset_meta_path: str = "training_tokens.meta.json"
    validation_split: float = 0.10
    seed: int = 1337

    # Model
    vocab_size: int = 256
    context_length: int = 256
    embedding_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    feed_forward_dim: int = 1536
    dropout: float = 0.10
    bias: bool = True

    # Training
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    training_steps: int = 5000
    eval_interval: int = 100
    eval_steps: int = 20
    checkpoint_interval: int = 100
    gradient_clip: float = 1.0
    num_workers: int = 0

    # Files
    checkpoint_dir: str = "checkpoints"
    best_checkpoint: str = "checkpoints/best_model.pt"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Config":
        allowed = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in values.items() if key in allowed})


DEFAULT_CONFIG = Config()
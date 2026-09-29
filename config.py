"""Central configuration for the local subword GPT training project."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Config:
    # Data
    dataset_manifest_path: str = "dataset_manifest.json"
    dataset_source: str = "training_data.txt"
    tokenizer_path: str = "tokenizer.json"
    dataset_dtype: str = "uint16"
    seed: int = 1337

    # Model
    # Resolved from the loaded tokenizer at runtime. This target is used only
    # when training a new tokenizer in prepare_dataset.py.
    vocab_size: int = 0
    tokenizer_target_vocab_size: int = 4096
    context_length: int = 1024
    embedding_dim: int = 384
    num_layers: int = 8
    num_heads: int = 6
    feed_forward_dim: int = 1536
    dropout: float = 0.05
    bias: bool = True
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tokenizer_sha256: str = ""
    eos_token_id: int = 0
    pad_token_id: int = 0

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    training_steps: int = 20000
    eval_interval: int = 250
    eval_steps: int = 20
    checkpoint_interval: int = 1000
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
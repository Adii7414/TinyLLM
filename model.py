"""A small decoder-only GPT-style Transformer implemented directly in PyTorch."""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.query_key_value = nn.Linear(
            config.embedding_dim, 3 * config.embedding_dim, bias=config.bias
        )
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        q, k, v = self.query_key_value(x).split(channels, dim=2)
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:, :, :sequence_length, :sequence_length] == 0,
            torch.finfo(attention_scores.dtype).min,
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended = attention_weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.residual_dropout(self.output(attended))


class FeedForward(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization keeps optimization stable in deeper Transformers.
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.language_model_head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False
        )
        self.apply(self._initialize_weights)
        # Weight tying is common in language models and reduces redundant parameters.
        self.language_model_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length "
                f"{self.config.context_length}"
            )
        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        for block in self.blocks:
            x = block(x)
        logits = self.language_model_head(self.final_layer_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_count(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def estimated_memory_mb(self, bytes_per_parameter: int = 4) -> float:
        return self.parameter_count() * bytes_per_parameter / (1024**2)


def describe_model(model: GPTModel) -> str:
    config = model.config
    return "\n".join(
        [
            f"Model parameters: {model.parameter_count():,} "
            f"({model.parameter_count() / 1_000_000:.2f}M)",
            f"Estimated model memory (FP32 weights): {model.estimated_memory_mb():.2f} MB",
            f"Vocabulary: {config.vocab_size}",
            f"Context: {config.context_length}",
            f"Layers: {config.num_layers}",
            f"Heads: {config.num_heads}",
            f"Embedding: {config.embedding_dim}",
            f"Feed-forward dimension: {config.feed_forward_dim}",
            f"Dropout: {config.dropout}",
        ]
    )
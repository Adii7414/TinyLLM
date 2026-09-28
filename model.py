"""A modern, compact decoder-only Transformer implemented directly in PyTorch."""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from config import Config


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, context_length: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("Attention head dimension must be even for RoPE.")
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(context_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_length = q.size(-2)
        cos = self.cos[:, :, :sequence_length, :]
        sin = self.sin[:, :, :sequence_length, :]
        # Interleave the two halves so the operation matches the standard
        # rotary embedding definition while keeping the cache compact.
        cos = torch.cat((cos, cos), dim=-1).to(dtype=q.dtype)
        sin = torch.cat((sin, sin), dim=-1).to(dtype=q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


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
        self.attention_dropout = config.dropout
        self.rotary = RotaryEmbedding(self.head_dim, config.context_length, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        q, k, v = self.query_key_value(x).split(channels, dim=2)
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate_and_value = nn.Linear(
            config.embedding_dim, 2 * config.feed_forward_dim, bias=config.bias
        )
        self.output = nn.Linear(config.feed_forward_dim, config.embedding_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_and_value(x).chunk(2, dim=-1)
        return self.dropout(self.output(F.silu(gate) * value))


class TransformerBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.embedding_dim, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.embedding_dim, config.norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.embedding_dim, config.norm_eps)
        self.language_model_head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False
        )
        self.apply(self._initialize_weights)
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
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length "
                f"{self.config.context_length}"
            )
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.language_model_head(self.final_norm(x))
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
            f"Feed-forward: {config.feed_forward_dim} (SwiGLU)",
            f"Position encoding: RoPE (theta={config.rope_theta:g})",
            f"Normalization: RMSNorm",
            f"Dropout: {config.dropout}",
        ]
    )
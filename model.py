"""A modern, compact decoder-only Transformer implemented directly in PyTorch."""

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from config import Config


KeyValue = Tuple[torch.Tensor, torch.Tensor]
KeyValueCache = Sequence[Optional[KeyValue]]


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
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)
        positions = torch.arange(context_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_length = q.size(-2)
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=q.device)
        if position_ids.ndim != 1 or position_ids.numel() != sequence_length:
            raise ValueError("position_ids must be a 1D tensor matching the sequence length.")
        if torch.any(position_ids < 0):
            raise ValueError("RoPE position IDs must be non-negative.")

        position_ids = position_ids.to(device=self.cos.device, dtype=torch.long)
        if position_ids.numel() == 0:
            cos = self.cos[:, :, :0, :]
            sin = self.sin[:, :, :0, :]
        elif int(position_ids.max()) < self.cos.size(2):
            cos = self.cos.index_select(2, position_ids)
            sin = self.sin.index_select(2, position_ids)
        else:
            # Cached decoding can continue past the configured attention window.
            # The attention cache still limits the number of visible tokens; RoPE
            # positions remain absolute so relative positions stay consistent as
            # the window rolls forward.
            angles = torch.outer(
                position_ids.to(dtype=self.inverse_frequency.dtype),
                self.inverse_frequency,
            )
            cos = angles.cos()[None, None, :, :]
            sin = angles.sin()[None, None, :, :]
        cos = cos.to(device=q.device, dtype=q.dtype)
        sin = sin.to(device=q.device, dtype=q.dtype)
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

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KeyValue] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, KeyValue]]:
        batch_size, sequence_length, channels = x.shape
        q, k, v = self.query_key_value(x).split(channels, dim=2)
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        position_ids = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=x.device,
        )
        q, k = self.rotary(q, k, position_ids)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # With a cache, every key is from the past or the current token, so
            # an explicit causal mask is unnecessary. For a full sequence, the
            # fused causal path preserves the original training behavior.
            is_causal=past_key_value is None,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        output = self.output(attended)
        if use_cache:
            return output, (k, v)
        return output


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

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KeyValue] = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, KeyValue]]:
        attention_output = self.attention(
            self.attention_norm(x),
            past_key_value=past_key_value,
            position_offset=position_offset,
            use_cache=use_cache,
        )
        if use_cache:
            attention_output, present_key_value = attention_output
        x = x + attention_output
        x = x + self.feed_forward(self.feed_forward_norm(x))
        if use_cache:
            return x, present_key_value
        return x


class GPTModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.vocab_size < 1:
            raise ValueError(
                "config.vocab_size is unresolved. Load a tokenizer and use its "
                "vocab_size before constructing GPTModel."
            )
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
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        past_key_values: Optional[KeyValueCache] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[KeyValue, ...]],
    ]:
        _, sequence_length = token_ids.shape
        if sequence_length < 1:
            raise ValueError("token_ids must contain at least one token.")
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative.")
        if past_key_values is not None:
            if len(past_key_values) != len(self.blocks):
                raise ValueError(
                    f"Expected {len(self.blocks)} layer caches, "
                    f"received {len(past_key_values)}."
                )
            cache_lengths = {
                key_value[0].size(2)
                for key_value in past_key_values
                if key_value is not None
            }
            if len(cache_lengths) > 1:
                raise ValueError("All layer caches must have the same sequence length.")
            past_length = next(iter(cache_lengths), 0)
        else:
            past_length = 0
        if past_length + sequence_length > self.config.context_length:
            raise ValueError(
                f"Cached sequence length {past_length + sequence_length} exceeds context length "
                f"{self.config.context_length}"
            )
        x = self.token_embedding(token_ids)
        present_key_values: List[KeyValue] = []
        for index, block in enumerate(self.blocks):
            past_key_value = (
                None if past_key_values is None else past_key_values[index]
            )
            block_output = block(
                x,
                past_key_value=past_key_value,
                position_offset=position_offset,
                use_cache=use_cache,
            )
            if use_cache:
                x, present_key_value = block_output
                present_key_values.append(present_key_value)
            else:
                x = block_output
        logits = self.language_model_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if use_cache:
            return logits, loss, tuple(present_key_values)
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
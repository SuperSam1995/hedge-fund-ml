"""Reusable neural network building blocks.

Currently holds light-weight Transformer blocks that operate on "series-as-tokens"
representations for time-series modelling.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - import guard mirrors optional dependency pattern
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - surfaced during import time
    msg = "PyTorch is required for hedge_fund_ml.models.blocks"
    raise ImportError(msg) from exc


@dataclass(frozen=True, slots=True)
class SeriesTransformerConfig:
    """Configuration for a Transformer block operating on series tokens."""

    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            msg = "embed_dim must be positive"
            raise ValueError(msg)
        if self.num_heads <= 0:
            msg = "num_heads must be positive"
            raise ValueError(msg)
        if self.embed_dim % self.num_heads != 0:
            msg = "embed_dim must be divisible by num_heads"
            raise ValueError(msg)
        if self.mlp_ratio <= 0:
            msg = "mlp_ratio must be positive"
            raise ValueError(msg)
        if self.dropout < 0:
            msg = "dropout cannot be negative"
            raise ValueError(msg)


class SeriesTransformerBlock(nn.Module):
    """Pre-norm Transformer block for series token interactions."""

    def __init__(self, config: SeriesTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and MLP mixing to the tokens."""

        residual = tokens
        tokens = self.norm1(tokens)
        attn_output, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = residual + self.dropout(attn_output)

        residual = tokens
        tokens = self.norm2(tokens)
        tokens = residual + self.mlp(tokens)
        return tokens


__all__ = ["SeriesTransformerConfig", "SeriesTransformerBlock"]

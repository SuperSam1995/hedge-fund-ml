"""iTransformer: treat time-series features as tokens for cross-sectional attention."""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - optional dependency guard
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - surfaced when torch missing
    msg = "PyTorch is required for the iTransformer module"
    raise ImportError(msg) from exc

from hedge_fund_ml.models.blocks import SeriesTransformerBlock, SeriesTransformerConfig


@dataclass(frozen=True, slots=True)
class ITransformerConfig:
    """Architecture configuration for the iTransformer model."""

    input_dim: int
    seq_len: int
    target_dim: int
    embed_dim: int = 64
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            msg = "input_dim must be positive"
            raise ValueError(msg)
        if self.seq_len <= 0:
            msg = "seq_len must be positive"
            raise ValueError(msg)
        if self.target_dim <= 0:
            msg = "target_dim must be positive"
            raise ValueError(msg)
        if self.depth <= 0:
            msg = "depth must be positive"
            raise ValueError(msg)
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


class ITransformer(nn.Module):
    """Cross-sectional Transformer leveraging series-as-token representations."""

    def __init__(self, config: ITransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_projection = nn.Linear(config.seq_len, config.embed_dim)
        block_config = SeriesTransformerConfig(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            SeriesTransformerBlock(block_config) for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)

        self.head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.target_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the iTransformer forward pass.

        Parameters
        ----------
        inputs:
            Tensor of shape ``(batch_size, seq_len, input_dim)`` containing the
            time-series batch.
        """

        if inputs.ndim != 3:
            msg = "inputs must have shape (batch, seq_len, input_dim)"
            raise ValueError(msg)

        batch, seq_len, input_dim = inputs.shape
        if seq_len != self.config.seq_len:
            msg = f"Expected seq_len {self.config.seq_len}, received {seq_len}"
            raise ValueError(msg)
        if input_dim != self.config.input_dim:
            msg = f"Expected input_dim {self.config.input_dim}, received {input_dim}"
            raise ValueError(msg)

        tokens = inputs.transpose(1, 2)
        tokens = self.token_projection(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        pooled = tokens.mean(dim=1)
        outputs = self.head(pooled)
        return outputs


__all__ = ["ITransformer", "ITransformerConfig"]

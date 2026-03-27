"""Smoke tests for the iTransformer module."""

from __future__ import annotations

import pytest
import torch

from hedge_fund_ml.models.itransformer import ITransformer, ITransformerConfig


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    torch.manual_seed(0)


def test_itransformer_forward_shape() -> None:
    config = ITransformerConfig(
        input_dim=5,
        seq_len=12,
        target_dim=3,
        embed_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )

    model = ITransformer(config)
    batch = 7
    dummy_input = torch.randn(batch, config.seq_len, config.input_dim)
    output = model(dummy_input)

    assert output.shape == (batch, config.target_dim)
    assert torch.isfinite(output).all()


def test_itransformer_requires_valid_shapes() -> None:
    config = ITransformerConfig(input_dim=4, seq_len=10, target_dim=2)
    model = ITransformer(config)
    invalid = torch.randn(3, config.seq_len + 1, config.input_dim)

    with pytest.raises(ValueError, match="seq_len"):
        model(invalid)

    invalid_dim = torch.randn(3, config.seq_len, config.input_dim + 1)
    with pytest.raises(ValueError, match="input_dim"):
        model(invalid_dim)


def test_itransformer_config_validation() -> None:
    with pytest.raises(ValueError):
        ITransformerConfig(input_dim=0, seq_len=1, target_dim=1)

    with pytest.raises(ValueError):
        ITransformerConfig(input_dim=1, seq_len=1, target_dim=1, num_heads=3, embed_dim=8)

    config = ITransformerConfig(input_dim=2, seq_len=4, target_dim=1)
    assert config.embed_dim == 64
    assert config.depth == 2


def test_series_transformer_block_shapes() -> None:
    from hedge_fund_ml.models.blocks import SeriesTransformerBlock, SeriesTransformerConfig

    config = SeriesTransformerConfig(embed_dim=8, num_heads=2, mlp_ratio=2.0, dropout=0.0)
    block = SeriesTransformerBlock(config)
    tokens = torch.randn(2, 5, config.embed_dim)
    output = block(tokens)

    assert output.shape == tokens.shape
    assert torch.isfinite(output).all()


def test_series_transformer_config_validation() -> None:
    from hedge_fund_ml.models.blocks import SeriesTransformerConfig

    with pytest.raises(ValueError):
        SeriesTransformerConfig(embed_dim=0, num_heads=1)
    with pytest.raises(ValueError):
        SeriesTransformerConfig(embed_dim=8, num_heads=3)
    with pytest.raises(ValueError):
        SeriesTransformerConfig(embed_dim=8, num_heads=2, dropout=-0.1)


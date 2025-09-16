"""Tests for return-based feature generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import ReturnsBuilder, ReturnsConfig


def test_returns_builder_shapes_and_shift() -> None:
    index = pd.date_range("2020-01-31", periods=10, freq="ME")
    base = pd.DataFrame({"fund": np.linspace(0.01, 0.1, len(index))}, index=index)

    config = ReturnsConfig(horizon=1, lag_periods=[1, 2], source="returns", dropna=True)
    dataset = ReturnsBuilder(config).build(features=base, target=base)

    max_lag = max(config.lag_periods)
    expected_index = index[max_lag : len(index) - config.horizon]
    assert dataset.features.index.equals(expected_index)
    assert dataset.target.index.equals(expected_index)

    # Lagged features rely only on past information.
    for lag in config.lag_periods:
        expected = base.shift(lag).loc[expected_index, "fund"]
        lag_slice = dataset.features.xs(f"lag_{lag}", axis=1, level=1)
        if isinstance(lag_slice, pd.DataFrame):
            result = lag_slice.loc[:, "fund"]
        else:
            result = lag_slice
        np.testing.assert_allclose(result.to_numpy(), expected.to_numpy())

    # Targets are shifted forward by the prediction horizon.
    expected_target = base.shift(-config.horizon).loc[expected_index, "fund"]
    horizon_slice = dataset.target.xs(f"fwd_{config.horizon}", axis=1, level=1)
    if isinstance(horizon_slice, pd.DataFrame):
        result_target = horizon_slice.loc[:, "fund"]
    else:
        result_target = horizon_slice
    np.testing.assert_allclose(result_target.to_numpy(), expected_target.to_numpy())

    # One feature column per lag.
    assert dataset.features.shape[1] == len(config.lag_periods)

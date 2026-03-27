from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data.windowing import (
    WindowingConfig,
    ZScoreScaler,
    load_panels,
    make_windows,
)


@pytest.fixture(scope="module")
def windowing_config() -> WindowingConfig:
    payload = yaml.safe_load(Path("configs/transformer.yaml").read_text())
    subset = {key: payload[key] for key in ("data", "split")}
    return WindowingConfig.model_validate(subset)


def test_load_panels_month_end(windowing_config: WindowingConfig) -> None:
    X, y = load_panels(windowing_config.data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert X.index.equals(y.index)
    assert (X.index == X.index.to_period("M").to_timestamp("M")).all()
    assert not X.isna().any().any()
    assert not y.isna().any().any()


def test_make_windows_alignment(windowing_config: WindowingConfig) -> None:
    cfg = windowing_config.data
    X, y = load_panels(cfg)
    dataset = make_windows(X, y, cfg.lookback, cfg.horizon)

    expected_windows = len(X) - cfg.lookback - cfg.horizon + 1
    assert dataset.features.shape == (expected_windows, cfg.lookback, X.shape[1])
    assert dataset.targets.shape == (expected_windows, len(cfg.targets))
    assert dataset.feature_index.equals(X.index[cfg.lookback - 1 : -cfg.horizon])
    assert dataset.target_index.equals(y.index[cfg.lookback + cfg.horizon - 1 :])

    offset = pd.offsets.MonthEnd(cfg.horizon)
    assert all(t + offset == s for t, s in zip(dataset.feature_index, dataset.target_index, strict=False))


def test_zscore_scaler_train_only(windowing_config: WindowingConfig) -> None:
    cfg = windowing_config
    X, y = load_panels(cfg.data)
    dataset = make_windows(X, y, cfg.data.lookback, cfg.data.horizon)

    train_mask = dataset.target_index <= cfg.split.train_end_ts
    train_features = dataset.features[train_mask]
    scaler = ZScoreScaler.fit(train_features)

    transformed_train = scaler.transform(train_features)
    mean = transformed_train.mean(axis=(0, 1))
    std = transformed_train.std(axis=(0, 1))

    assert np.allclose(mean, 0.0, atol=1e-8)
    assert np.allclose(std, 1.0, atol=1e-6)

    transformed_all = scaler.transform(dataset.features)
    assert transformed_all.shape == dataset.features.shape

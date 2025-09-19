from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from report import plot_cumulative, plot_rolling_te, plot_turnover


def _make_returns_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=48, freq="ME", name="date")
    rng = np.random.default_rng(7)
    columns = pd.MultiIndex.from_product(
        [["target", "replication"], ["return"]],
        names=["series", "metric"],
    )
    target = rng.normal(0.005, 0.01, size=len(index))
    replica = target + rng.normal(0.0, 0.003, size=len(index))
    values = np.column_stack([target, replica])
    return pd.DataFrame(values, index=index, columns=columns)


def _make_weights_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=12, freq="ME", name="date")
    columns = pd.MultiIndex.from_product(
        [["replication"], ["asset_a", "asset_b"]],
        names=["strategy", "asset"],
    )
    data = np.linspace(0.2, 0.8, num=len(index) * len(columns)).reshape(len(index), len(columns))
    return pd.DataFrame(data, index=index, columns=columns)


def test_plot_cumulative_creates_expected_file(tmp_path: Path) -> None:
    frame = _make_returns_frame()
    path = plot_cumulative(frame, "replication", tmp_path)
    assert path.exists()
    assert path.name == "replication_cumulative.png"
    assert path.stat().st_size > 0


def test_plot_rolling_te_creates_expected_file(tmp_path: Path) -> None:
    frame = _make_returns_frame()
    path = plot_rolling_te(frame, "replication", window=12, out_dir=tmp_path)
    assert path.exists()
    assert path.name == "replication_rolling_te_12.png"
    assert path.stat().st_size > 0


def test_plot_turnover_creates_expected_file(tmp_path: Path) -> None:
    weights = _make_weights_frame()
    path = plot_turnover(weights, "replication", tmp_path)
    assert path.exists()
    assert path.name == "replication_turnover.png"
    assert path.stat().st_size > 0

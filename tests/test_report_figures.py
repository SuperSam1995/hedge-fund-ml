from __future__ import annotations

from pathlib import Path

import pandas as pd

from report import plot_cum, plot_roll_te


def _make_returns_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=48, freq="ME", name="date")
    returns = []
    for date in index:
        returns.extend(
            [
                {
                    "date": date,
                    "strategy": "replication",
                    "role": "target",
                    "return": 0.01,
                },
                {
                    "date": date,
                    "strategy": "replication",
                    "role": "replica",
                    "return": 0.009,
                },
            ]
        )
    return pd.DataFrame(returns)


def test_plot_cumulative_creates_expected_file(tmp_path: Path) -> None:
    frame = _make_returns_frame()
    plot_cum(frame, "replication", tmp_path)
    path = tmp_path / "cum_replication.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_rolling_te_creates_expected_file(tmp_path: Path) -> None:
    frame = _make_returns_frame()
    plot_roll_te(frame, "replication", tmp_path, win=12)
    path = tmp_path / "rollTE_replication.png"
    assert path.exists()
    assert path.stat().st_size > 0

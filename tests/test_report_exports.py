from __future__ import annotations

import pandas as pd

from report import export_metrics_long, export_returns_long, export_weights_long


def test_export_metrics_long(tmp_path):
    frame = pd.DataFrame(
        {
            "strategy": ["alpha", "alpha", "beta", "beta"],
            "role": ["replica", "target", "replica", "target"],
            "ann_return": [0.1, 0.2, 0.3, 0.4],
            "ann_vol": [0.5, 0.6, 0.7, 0.8],
        }
    )

    export_metrics_long(frame, tmp_path)

    exported = pd.read_csv(tmp_path / "metrics_by_series.csv")
    pd.testing.assert_frame_equal(exported, frame)


def test_export_returns_long(tmp_path):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "strategy": ["alpha", "alpha", "beta", "beta"],
            "role": ["target", "replica", "target", "replica"],
            "return": [0.01, 0.02, 0.03, 0.04],
        }
    )

    export_returns_long(frame, tmp_path)

    exported = pd.read_csv(tmp_path / "returns_long.csv", parse_dates=["date"])
    pd.testing.assert_frame_equal(exported, frame)


def test_export_weights_long(tmp_path):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "strategy": ["alpha", "alpha", "beta", "beta"],
            "ticker": ["A", "B", "A", "B"],
            "weight": [0.1, 0.2, 0.3, 0.4],
        }
    )

    export_weights_long(frame, tmp_path, ["alpha", "beta"])

    alpha = pd.read_csv(tmp_path / "weights_alpha.csv", parse_dates=["date"])
    beta = pd.read_csv(tmp_path / "weights_beta.csv", parse_dates=["date"])

    pd.testing.assert_frame_equal(
        alpha.reset_index(drop=True), frame[frame["strategy"] == "alpha"].reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        beta.reset_index(drop=True), frame[frame["strategy"] == "beta"].reset_index(drop=True)
    )

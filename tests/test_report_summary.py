from __future__ import annotations

import pandas as pd
import pytest

from report import build_metrics_summary


def test_build_metrics_summary_basic():
    metrics_long = pd.DataFrame(
        {
            "strategy": ["alpha", "alpha", "beta", "beta"],
            "role": ["replica", "target", "replica", "target"],
            "ann_return": [0.12, 0.15, 0.1, 0.08],
        }
    )

    summary = build_metrics_summary(metrics_long)

    assert set(summary.keys()) == {"alpha", "beta"}
    assert summary["alpha"]["replica"]["ann_return"] == 0.12
    assert summary["alpha"]["target"]["ann_return"] == 0.15


def test_build_metrics_summary_handles_missing_roles():
    metrics_long = pd.DataFrame(
        {
            "strategy": ["alpha"],
            "role": ["replica"],
            "ann_return": [0.12],
        }
    )

    with pytest.raises(IndexError):
        build_metrics_summary(metrics_long)


def test_build_metrics_summary_empty_frame():
    metrics_long = pd.DataFrame(columns=["strategy", "role", "ann_return"])

    summary = build_metrics_summary(metrics_long)

    assert summary == {}

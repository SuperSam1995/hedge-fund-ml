from __future__ import annotations

import pandas as pd

from report import build_metrics_summary


def test_build_metrics_summary_basic():
    metrics_long = pd.DataFrame(
        {
            "series": ["alpha", "alpha", "beta", "beta", "gamma", "gamma"],
            "metric": [
                "ann_return",
                "ann_vol",
                "ann_return",
                "ann_vol",
                "ann_return",
                "ann_vol",
            ],
            "value": [0.12, 0.08, 0.1, 0.07, 0.15, 0.09],
        }
    )

    summary = build_metrics_summary(metrics_long)

    ann_return = summary["metrics"]["ann_return"]
    assert ann_return["best"] == {"series": "gamma", "value": 0.15}
    assert ann_return["worst"] == {"series": "beta", "value": 0.1}
    assert ann_return["count"] == 3
    assert summary["series"]["alpha"]["ann_vol"] == 0.08


def test_build_metrics_summary_handles_missing_columns():
    metrics_long = pd.DataFrame({"series": ["a"], "metric": ["ret"]})

    try:
        build_metrics_summary(metrics_long)
    except ValueError as exc:
        assert "metrics_long is missing required columns" in str(exc)
    else:  # pragma: no cover - fail loudly if no error is raised
        raise AssertionError("Expected ValueError when value column missing")


def test_build_metrics_summary_empty_frame():
    metrics_long = pd.DataFrame(columns=["series", "metric", "value"])

    summary = build_metrics_summary(metrics_long)

    assert summary["metrics"] == {}
    assert summary["series"] == {}
    assert "generated_at" in summary

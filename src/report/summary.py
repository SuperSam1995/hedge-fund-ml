"""Summary helpers for evaluation metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["build_metrics_summary"]


def _format_title(text: str) -> str:
    cleaned = str(text).replace("_", " ").replace("-", " ").strip()
    return cleaned.title() if cleaned else "Value"


def build_metrics_summary(metrics_long: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON serialisable metrics summary from long-form data.

    Parameters
    ----------
    metrics_long:
        DataFrame with ``series``, ``metric`` and ``value`` columns.

    Returns
    -------
    dict[str, Any]
        Nested mapping describing the distribution of each metric across
        evaluated series together with a per-series metric lookup.
    """

    required = {"series", "metric", "value"}
    missing = required.difference(metrics_long.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"metrics_long is missing required columns: {joined}")

    if metrics_long.empty:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "series": {},
        }

    tidy = metrics_long.copy()
    tidy = tidy.loc[:, ["series", "metric", "value"]]
    tidy["series"] = tidy["series"].astype(str)
    tidy["metric"] = tidy["metric"].astype(str)
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["value"])

    if tidy.empty:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "series": {},
        }

    metrics_summary: dict[str, Any] = {}
    series_summary: dict[str, dict[str, float]] = {}

    for metric, group in tidy.groupby("metric", sort=False):
        values = group["value"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        ordering = group.assign(__value=values).sort_values(
            "__value", ascending=False, kind="mergesort"
        )
        best = ordering.iloc[0]
        worst = ordering.iloc[-1]
        metrics_summary[str(metric)] = {
            "title": _format_title(metric),
            "best": {
                "series": str(best["series"]),
                "value": float(best["value"]),
            },
            "worst": {
                "series": str(worst["series"]),
                "value": float(worst["value"]),
            },
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=0)),
            "count": int(values.size),
        }

    for series, group in tidy.groupby("series", sort=False):
        series_summary[str(series)] = {
            str(metric): float(value)
            for metric, value in zip(group["metric"], group["value"], strict=False)
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics_summary,
        "series": series_summary,
    }

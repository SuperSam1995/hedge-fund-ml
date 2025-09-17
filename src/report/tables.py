"""Tabular reporting helpers for replication experiments."""

from __future__ import annotations

import pandas as pd

__all__ = ["metrics_table"]


def metrics_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert a nested metrics mapping into a tidy DataFrame."""

    frame = pd.DataFrame(metrics).T
    frame.index.name = "series"
    return frame.sort_index(axis=0)

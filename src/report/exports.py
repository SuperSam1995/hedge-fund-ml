from pathlib import Path

import pandas as pd


def export_metrics_long(df_metrics: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out / "metrics_by_series.csv", index=False)


def export_returns_long(df_series: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df_series.to_csv(out / "returns_long.csv", index=False)


def export_weights_long(df_weights: pd.DataFrame, out: Path, strategies) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for s in strategies:
        sub = df_weights[df_weights["strategy"] == s]
        sub.to_csv(out / f"weights_{s}.csv", index=False)

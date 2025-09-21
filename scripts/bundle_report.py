import argparse
import datetime
import subprocess
from pathlib import Path

import pandas as pd

from report import (
    build_metrics_summary,
    export_metrics_long,
    export_returns_long,
    export_weights_long,
    plot_cum,
    plot_roll_te,
    write_json,
)


def sh(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def main(out_dir: str):
    run_id = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H%MZ")
    root = Path(out_dir or f"reports/bundle/{run_id}")
    figs = root / "figures"
    tabs = root / "tables"
    root.mkdir(parents=True, exist_ok=True)

    metrics_src = Path("reports/metrics/replication_metrics.csv")
    series_src = Path("reports/series/replication_series.csv")
    weights_src = Path("data/interim/replication_weights.csv")

    for path in (metrics_src, series_src, weights_src):
        if not path.exists():
            raise FileNotFoundError(f"Required artefact missing: {path}")

    metrics = pd.read_csv(metrics_src)
    series = pd.read_csv(series_src, parse_dates=["date"])
    try:
        weights = pd.read_csv(weights_src, parse_dates=["date"])
    except pd.errors.EmptyDataError:
        weights = pd.DataFrame(columns=["date", "strategy", "ticker", "weight"])
    strategies = sorted(series["strategy"].unique())

    export_metrics_long(metrics, tabs)
    export_returns_long(series, tabs)
    export_weights_long(weights, tabs, strategies)

    for s in strategies:
        plot_cum(series, s, figs)
        plot_roll_te(series, s, figs)

    # metadata
    ctx = {
        "python": sh("python -V"),
        "git_commit": (
            sh("git rev-parse --short HEAD") if (Path(".git") / "HEAD").exists() else "zip"
        ),
        "run_date": run_id,
        "strategies": strategies,
        "paths": {
            "panel_csv": "cleaned_data/factor_etf_data.csv",
            "target_csv": "cleaned_data/hfd.csv",
        },
    }
    write_json(ctx, root / "context.json")
    summary = build_metrics_summary(metrics)
    write_json(summary, root / "metrics_summary.json")

    manifest = {
        "run_id": run_id,
        "figures": (
            [{"id": f"cum_{s}", "path": f"figures/cum_{s}.png"} for s in strategies]
            + [{"id": f"rollTE_{s}", "path": f"figures/rollTE_{s}.png"} for s in strategies]
        ),
        "tables": (
            [
                {"id": "metrics_by_series", "path": "tables/metrics_by_series.csv"},
                {"id": "returns_long", "path": "tables/returns_long.csv"},
            ]
            + [{"id": f"weights_{s}", "path": f"tables/weights_{s}.csv"} for s in strategies]
        ),
        "paths": ctx["paths"],
    }
    write_json(manifest, root / "manifest.json")

    # index.md scaffold
    figures_listing = "\n".join(
        f"- {s}: [cum](figures/cum_{s}.png), [rollTE](figures/rollTE_{s}.png)" for s in strategies
    )
    data_line = (
        f"Inputs: {ctx['paths']['panel_csv']} (features), "
        f"{ctx['paths']['target_csv']} (targets). Monthly frequency."
    )

    (root / "index.md").write_text(
        f"""---
title: Hedge Fund Strategy Replication — Run {run_id}
date: {run_id}
commit: {ctx['git_commit']}
---

# 1. Introduction
(LLM: summarize top-line results from metrics_summary.json)

# 2. Data
{data_line}

# 3. Methodology
3.1 Baseline (HK span + OLS). (LLM: describe from repo configs.)
3.2 Proposed (if present). (LLM: describe variant models.)

# 4. Results
- Metrics table: [tables/metrics_by_series.csv](tables/metrics_by_series.csv)

## Figures
{figures_listing}

# 5. Robustness
(LLM: discuss costs on/off, seed stability if tables present.)

# 6. Conclusion
(LLM: synthesize wins & gaps.)
"""
    )
    print(f"Bundle -> {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    main(args.out)

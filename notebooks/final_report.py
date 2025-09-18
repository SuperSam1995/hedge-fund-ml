# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Final Performance Report
#
# This notebook assembles the latest evaluation metrics and curated figures into a publishable report.

# %% tags=["parameters"]
# Parameters
metrics_path = "reports/metrics_latest.json"
figures_dir = "reports/figures"

# %%
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from IPython.display import Image, Markdown, display

metrics_path = Path(metrics_path)
figures_dir = Path(figures_dir)
report_generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

if not metrics_path.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

figures_dir.mkdir(parents=True, exist_ok=True)


# %%
def _load_metrics(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.json_normalize(data).T.rename(columns={0: "value"})
        raise TypeError(f"Unsupported JSON structure: {type(data)}")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metrics format: {suffix}")

metrics_df = _load_metrics(metrics_path)
metrics_df.index.name = metrics_df.index.name or "metric"
metrics_df.columns = [str(col) for col in metrics_df.columns]

display(Markdown("## Key Metrics"))
display(metrics_df)


# %%
def _iter_figures(directory: Path) -> Iterable[Path]:
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
        yield from sorted(directory.glob(suffix))

figure_paths = list(_iter_figures(figures_dir))
if figure_paths:
    display(Markdown("## Figures"))
    for figure_path in figure_paths:
        title = figure_path.stem.replace("_", " ").title()
        display(Markdown(f"### {title}"))
        display(Image(filename=str(figure_path)))
else:
    display(Markdown("## Figures\n\n_No figures available for this run._"))


# %%
display(Markdown(f"Report generated at **{report_generated_at}**"))

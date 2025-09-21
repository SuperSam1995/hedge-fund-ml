import json
from pathlib import Path

import pandas as pd


def build_metrics_summary(metrics_long: pd.DataFrame):
    out = {}
    for s in metrics_long["strategy"].unique():
        sub = metrics_long[metrics_long["strategy"] == s]
        rep = sub[sub["role"] == "replica"].iloc[0].to_dict()
        tgt = sub[sub["role"] == "target"].iloc[0].to_dict()
        out[s] = {"replica": rep, "target": tgt}
    return out


def write_json(d, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2))

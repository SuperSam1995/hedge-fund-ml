from __future__ import annotations

import pandas as pd

from report import (
    export_metrics_long,
    export_returns_long,
    export_weights_long,
)


def test_export_metrics_long(tmp_path):
    frame = pd.DataFrame(
        {
            "ann_return": [0.1, 0.2],
            "ann_vol": [0.3, 0.4],
        },
        index=pd.Index(["target:a", "replica:a"], name="series"),
    )

    output = export_metrics_long(frame, tmp_path)

    exported = pd.read_csv(output)
    assert set(exported.columns) == {"series", "metric", "value"}
    assert len(exported) == 4


def test_export_returns_long(tmp_path):
    columns = pd.MultiIndex.from_product(
        [["target", "replica"], ["asset_a", "asset_b"]],
        names=["series", "asset"],
    )
    index = pd.date_range("2024-01-01", periods=2, freq="D", name="date")
    frame = pd.DataFrame(
        [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08]],
        index=index,
        columns=columns,
    )

    output = export_returns_long(frame, tmp_path)

    exported = pd.read_csv(output)
    assert exported.columns.tolist() == ["date", "series", "asset", "return"]
    assert exported.shape[0] == 8


def test_export_weights_long(tmp_path):
    columns = pd.MultiIndex.from_product(
        [["alpha", "beta"], ["asset_a", "asset_b"]],
        names=["strategy", "asset"],
    )
    index = pd.date_range("2024-01-01", periods=2, freq="D", name="date")
    frame = pd.DataFrame(
        [[0.5, 0.5, 0.4, 0.6], [0.6, 0.4, 0.3, 0.7]],
        index=index,
        columns=columns,
    )

    outputs = export_weights_long(frame, tmp_path)

    assert set(outputs.keys()) == {"alpha", "beta"}
    for path in outputs.values():
        exported = pd.read_csv(path)
        assert exported.columns.tolist() == ["date", "asset", "weight"]
        assert exported.shape[0] == 4

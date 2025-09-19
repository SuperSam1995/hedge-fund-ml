from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from replicate import ITrafoConfig, run_itrafo_replication
from replicate.decoder import WeightDecoder


def test_weight_decoder_closed_form() -> None:
    rhat = np.array([0.01, 0.02], dtype=float)
    target = 0.05
    ridge = 1e-3
    decoder = WeightDecoder(leverage=10.0, lambda_to=0.0, lambda_l2=ridge, long_only=False)
    result = decoder.solve_once(rhat, target)

    expected = target * rhat / (np.dot(rhat, rhat) + ridge)
    np.testing.assert_allclose(result.weights, expected, atol=1e-6)


def test_run_itrafo_replication(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="M", tz="UTC")
    forecast = pd.DataFrame(
        {
            "date": dates,
            "strategy": ["A"] * 3,
            "yhat": [0.01, 0.02, 0.015],
        }
    )
    etf_forecast = pd.DataFrame(
        {
            "date": dates,
            "ETF1": [0.01, 0.015, 0.012],
            "ETF2": [0.005, 0.007, 0.006],
        }
    )

    itrafo_path = tmp_path / "itrafo.csv"
    etf_path = tmp_path / "etf.csv"
    weights_path = tmp_path / "weights.csv"
    series_path = tmp_path / "series.csv"
    metadata_path = tmp_path / "meta.json"

    forecast.to_csv(itrafo_path, index=False)
    etf_forecast.to_csv(etf_path, index=False)

    config_payload = {
        "seed": 7,
        "paths": {
            "itrafo_forecast_csv": str(itrafo_path),
            "etf_forecast_csv": str(etf_path),
            "weights_csv": str(weights_path),
            "series_csv": str(series_path),
            "metadata_json": str(metadata_path),
        },
        "cols": {
            "date": "date",
            "strategy": "strategy",
            "yhat": "yhat",
            "etfs": ["ETF1", "ETF2"],
        },
        "hyper": {
            "leverage": 1.0,
            "lambda_to": 0.01,
            "lambda_l2": 0.001,
            "long_only": False,
            "solver": "SCS",
            "solver_opts": {"eps": 1e-5, "max_iters": 5_000},
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    config = ITrafoConfig.from_yaml(config_path)
    result = run_itrafo_replication(config)

    assert weights_path.exists()
    assert series_path.exists()
    assert metadata_path.exists()

    assert not result.weights.empty
    assert not result.series.empty
    assert set(result.weights.columns) == {"date", "strategy", "ETF1", "ETF2"}
    assert set(result.series.columns) == {
        "date",
        "strategy",
        "portfolio_return_hat",
        "target_return_hat",
    }

    leverage = np.abs(result.weights[["ETF1", "ETF2"]]).sum(axis=1)
    assert (leverage <= config.hyper.leverage + 1e-6).all()

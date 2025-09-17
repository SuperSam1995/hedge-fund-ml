from __future__ import annotations

import numpy as np
import pandas as pd

from eval.metrics import (
    annualised_return,
    annualised_volatility,
    certainty_equivalent,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)


def test_basic_metrics() -> None:
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    returns = pd.Series(0.01, index=dates)

    assert np.isclose(
        annualised_return(returns, periods_per_year=12), 0.12682503013196977
    )
    assert np.isclose(
        annualised_volatility(returns, periods_per_year=12), 0.0
    )
    assert np.isnan(sharpe_ratio(returns, periods_per_year=12))
    assert np.isnan(sortino_ratio(returns, periods_per_year=12))
    assert np.isclose(max_drawdown(returns), 0.0)
    assert np.isclose(
        certainty_equivalent(returns, risk_aversion=3.0, periods_per_year=12), 0.12
    )
    assert np.isinf(omega_ratio(returns))


def test_turnover() -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    weights = pd.DataFrame(
        {
            "asset_a": [0.2, 0.3, 0.1],
            "asset_b": [0.8, 0.7, 0.9],
        },
        index=dates,
    )
    assert np.isclose(turnover(weights), 0.15)

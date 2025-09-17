"""Portfolio evaluation metrics with annualisation controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "annualised_return",
    "annualised_volatility",
    "certainty_equivalent",
    "max_drawdown",
    "omega_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "turnover",
]


def _ensure_series(data: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Expected a single column when passing a DataFrame")
        return data.iloc[:, 0]
    return data


def _dropna(data: pd.Series) -> pd.Series:
    cleaned = data.dropna()
    if cleaned.empty:
        raise ValueError("Series must contain at least one non-NaN observation")
    return cleaned


def annualised_return(returns: pd.Series | pd.DataFrame, periods_per_year: int) -> float:
    series = _dropna(_ensure_series(returns))
    compounded = (1 + series).prod()
    n_periods = len(series)
    if n_periods == 0:
        raise ValueError("No observations available for annualised return")
    return compounded ** (periods_per_year / n_periods) - 1


def annualised_volatility(returns: pd.Series | pd.DataFrame, periods_per_year: int) -> float:
    series = _dropna(_ensure_series(returns))
    return float(series.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    series = _dropna(_ensure_series(returns))
    excess = series - risk_free_rate / periods_per_year
    vol = excess.std(ddof=0)
    if vol <= np.finfo(float).eps:
        return np.nan
    return float(excess.mean() * periods_per_year / (vol * np.sqrt(periods_per_year)))


def sortino_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    series = _dropna(_ensure_series(returns))
    excess = series - risk_free_rate / periods_per_year
    downside = excess.copy()
    downside[downside > 0] = 0
    downside_std = downside.std(ddof=0)
    if downside_std <= np.finfo(float).eps:
        return np.nan
    return float(excess.mean() * periods_per_year / (abs(downside_std) * np.sqrt(periods_per_year)))


def max_drawdown(returns: pd.Series | pd.DataFrame) -> float:
    series = _dropna(_ensure_series(returns))
    cumulative = (1 + series).cumprod()
    peaks = cumulative.cummax()
    drawdowns = (cumulative - peaks) / peaks
    return float(drawdowns.min())


def turnover(weights: pd.DataFrame) -> float:
    if weights.empty:
        raise ValueError("weights must contain at least one observation")
    diffs = weights.diff().abs().sum(axis=1)
    if len(diffs) <= 1:
        return 0.0
    return float(0.5 * diffs.iloc[1:].mean())


def certainty_equivalent(
    returns: pd.Series | pd.DataFrame,
    risk_aversion: float = 3.0,
    periods_per_year: int = 12,
) -> float:
    series = _dropna(_ensure_series(returns))
    mean = series.mean() * periods_per_year
    variance = series.var(ddof=0) * periods_per_year
    return float(mean - 0.5 * risk_aversion * variance)


def omega_ratio(
    returns: pd.Series | pd.DataFrame,
    threshold: float = 0.0,
) -> float:
    series = _dropna(_ensure_series(returns))
    gains = (series - threshold).clip(lower=0)
    losses = (threshold - series).clip(lower=0)
    loss_sum = losses.sum()
    if loss_sum == 0:
        return np.inf
    return float(gains.sum() / loss_sum)

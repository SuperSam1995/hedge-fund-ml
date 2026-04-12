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


def _as_float_series(data: pd.Series | pd.DataFrame) -> pd.Series:
    series = _dropna(_ensure_series(data))
    numeric = series.to_numpy(dtype=float, copy=False)
    return pd.Series(numeric, index=series.index, dtype=float)


def annualised_return(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int,
) -> float:
    series = _as_float_series(returns)
    compounded = float(np.prod(1 + series.to_numpy()))
    n_periods = len(series)
    if n_periods == 0:
        raise ValueError("No observations available for annualised return")
    exponent = periods_per_year / float(n_periods)
    return float(compounded**exponent - 1.0)


def annualised_volatility(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int,
) -> float:
    series = _as_float_series(returns)
    return float(series.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    series = _as_float_series(returns)
    excess = series - risk_free_rate / periods_per_year
    vol = excess.std(ddof=0)
    if vol <= np.finfo(float).eps:
        return np.nan
    numerator = excess.mean() * periods_per_year
    denominator = vol * np.sqrt(periods_per_year)
    return float(numerator / denominator)


def sortino_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    series = _as_float_series(returns)
    excess = series.to_numpy() - risk_free_rate / periods_per_year
    # ⚡ Bolt Optimization: Use numpy clip instead of pandas series copying and boolean indexing
    downside = np.clip(excess, a_min=None, a_max=0)
    downside_std = downside.std(ddof=0)
    if downside_std <= np.finfo(float).eps:
        return np.nan
    numerator = float(excess.mean()) * periods_per_year
    denominator = abs(float(downside_std)) * np.sqrt(periods_per_year)
    return float(numerator / denominator)


def max_drawdown(returns: pd.Series | pd.DataFrame) -> float:
    series = _as_float_series(returns)
    cumulative_values = np.cumprod(1.0 + series.to_numpy())
    # ⚡ Bolt Optimization: Use numpy maximum accumulate instead of pandas cummax for max drawdown
    peaks = np.maximum.accumulate(cumulative_values)
    drawdowns = (cumulative_values - peaks) / peaks
    return float(np.min(drawdowns))


def turnover(weights: pd.DataFrame) -> float:
    if weights.empty:
        raise ValueError("weights must contain at least one observation")
    # ⚡ Bolt Optimization: Replace pandas .diff() with NumPy array operations for speed
    numeric = weights.to_numpy(dtype=float)
    if len(numeric) <= 1:
        return 0.0
    diffs = np.abs(np.diff(numeric, axis=0)).sum(axis=1)
    return float(0.5 * np.mean(diffs))


def certainty_equivalent(
    returns: pd.Series | pd.DataFrame,
    risk_aversion: float = 3.0,
    periods_per_year: int = 12,
) -> float:
    series = _as_float_series(returns)
    values = series.to_numpy()
    mean = float(np.mean(values)) * periods_per_year
    variance = float(np.var(values, ddof=0)) * periods_per_year
    return float(mean - 0.5 * risk_aversion * variance)


def omega_ratio(
    returns: pd.Series | pd.DataFrame,
    threshold: float = 0.0,
) -> float:
    series = _as_float_series(returns)
    # ⚡ Bolt Optimization: Use numpy clip instead of pandas series clip
    vals = series.to_numpy()
    gains = np.clip(vals - threshold, a_min=0, a_max=None)
    losses = np.clip(threshold - vals, a_min=0, a_max=None)
    loss_sum = losses.sum()
    if loss_sum == 0:
        return np.inf
    return float(gains.sum() / loss_sum)

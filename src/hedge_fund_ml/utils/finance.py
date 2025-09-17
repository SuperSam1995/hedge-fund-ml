"""Utility functions for portfolio analytics.

The original repository bundled a grab-bag of helpers without typing or
runtime validation.  The rewritten module keeps the research workflow intact
while providing deterministic behaviour, clearer error messages and type
annotations to support static analysis.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "normalization",
    "read_csv",
    "load_pickle",
    "save_pickle",
    "random_sampling",
    "transaction_cost",
    "price_impact",
    "reshape_cab",
    "ex_post_return",
    "factor_hf_split",
]


def _validate_window(window: int) -> None:
    if window < 2:
        raise ValueError("window must be at least 2 to compute sample variance")


def normalization(
    y: ArrayLike,
    x: ArrayLike,
    beta: ArrayLike,
    window: int,
) -> NDArray[np.float64]:
    """Return the normalisation factor described in the research notes.

    Parameters
    ----------
    y, x, beta
        Observed returns, factor matrix and regression coefficients.
    window
        Rolling window size used to scale the series.
    """

    _validate_window(window)
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)

    r_hat = x_arr @ beta_arr
    numerator = np.var(y_arr, axis=0, ddof=1)
    denominator = np.var(r_hat, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(numerator) / np.sqrt(denominator)
    result = np.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(result, dtype=np.float64)


def read_csv(path: Path | str, parse_dates: bool = True) -> pd.DataFrame:
    """Load a CSV file with optional date parsing."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, parse_dates=["Date"] if parse_dates else None)
    if parse_dates and "Date" in df:
        df = df.set_index("Date").sort_index()
    return df


def load_pickle(path: Path | str):
    """Read a pickle file from disk using ``pathlib`` semantics."""

    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_pickle(obj, path: Path | str) -> None:
    """Persist an object as a pickle file and verify readability."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file:
        pickle.dump(obj, file)
    # Minimal smoke test to ensure the file can be read back.
    load_pickle(target)


def random_sampling(
    dataset: ArrayLike,
    n_samples: int,
    window: int,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Return ``n_samples`` rolling windows drawn uniformly from ``dataset``."""

    data = np.asarray(dataset, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("dataset must be a 2D array-like")
    if window <= 0 or window > data.shape[0]:
        raise ValueError("window must be between 1 and len(dataset)")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    generator = rng or np.random.default_rng()
    indices = generator.integers(0, data.shape[0] - window + 1, size=n_samples)
    samples = np.stack([data[start : start + window] for start in indices])
    return cast(NDArray[np.float64], samples)


def transaction_cost(
    old_weights: ArrayLike,
    new_weights: ArrayLike,
    covariance: ArrayLike,
    impact_scale: float = 0.05,
) -> NDArray[np.float64]:
    """Quadratic transaction cost model from the original notebook."""

    if impact_scale < 0:
        raise ValueError("impact_scale must be non-negative")

    covariance_arr = np.asarray(covariance, dtype=np.float64)
    diag = np.sqrt(np.diag(covariance_arr)) * impact_scale
    delta = np.asarray(old_weights, dtype=np.float64) - np.asarray(
        new_weights, dtype=np.float64
    )
    cost = 0.5 * np.square(delta) * diag
    return cast(NDArray[np.float64], cost)


def price_impact(
    old_weights: ArrayLike,
    new_weights: ArrayLike,
    covariance: ArrayLike,
    impact_scale: float = 0.05,
    phi: float = 0.5,
) -> NDArray[np.float64]:
    """Simple linear-quadratic price impact approximation."""

    if impact_scale < 0 or phi < 0:
        raise ValueError("impact_scale and phi must be non-negative")

    covariance_arr = np.asarray(covariance, dtype=np.float64)
    diag = np.sqrt(np.diag(covariance_arr)) * impact_scale
    old_arr = np.asarray(old_weights, dtype=np.float64)
    new_arr = np.asarray(new_weights, dtype=np.float64)
    delta = old_arr - new_arr
    impact = (
        phi * new_arr * diag * delta
        - old_arr * diag * delta
        - 0.5 * np.square(delta) * diag
    )
    return cast(NDArray[np.float64], impact)


def reshape_cab(dataframes: Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
    """Convert a list of matrices with shape ``(A, B, C)`` to ``(C, A, B)``."""

    if not dataframes:
        raise ValueError("dataframes must contain at least one element")
    columns = dataframes[0].columns
    reshaped: List[pd.DataFrame] = []
    for column in columns:
        matrix = pd.DataFrame([df[column] for df in dataframes])
        reshaped.append(matrix)
    return reshaped


def ex_post_return(
    ex_ante: pd.DataFrame,
    window: int,
    strat_weight: Sequence[pd.DataFrame],
    factor_etf: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ex-post returns after deducting transaction costs and impact."""

    if window <= 0:
        raise ValueError("window must be positive")

    penalties: List[List[float]] = []
    for idx in range(len(ex_ante.columns)):
        series_penalties: List[float] = []
        for i in range(1, len(factor_etf) - window):
            cov_matrix = factor_etf.iloc[i : i + window].cov()
            new_x = strat_weight[idx].iloc[i]
            old_x = strat_weight[idx].iloc[i - 1]
            tc = transaction_cost(old_x, new_x, cov_matrix)
            pi = price_impact(old_x, new_x, cov_matrix)
            series_penalties.append(float(np.sum(tc + pi)))
        penalties.append(series_penalties)

    ex_post: List[List[float]] = []
    for idx, column in enumerate(ex_ante.columns):
        series: List[float] = [float(ex_ante.iloc[:, idx][0])]
        for i in range(1, len(ex_ante)):
            series.append(float(ex_ante.iloc[:, idx][i] + penalties[idx][i - 1]))
        ex_post.append(series)
    return pd.DataFrame(ex_post, columns=ex_ante.index, index=ex_ante.columns).T


def factor_hf_split(
    array: ArrayLike,
    split_pos: int,
    reshape: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a 3D array into factor and hedge-fund slices."""

    data = np.asarray(array, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("array must be 3-dimensional")
    if not 0 < split_pos < data.shape[2]:
        raise ValueError("split_pos must lie strictly within the last dimension")

    factors = data[:, :, :split_pos]
    hedge_funds = data[:, :, split_pos:]
    if reshape:
        factors = factors.reshape(factors.shape[0] * factors.shape[1], factors.shape[2])
        hedge_funds = hedge_funds.reshape(
            hedge_funds.shape[0] * hedge_funds.shape[1], hedge_funds.shape[2]
        )
    return factors, hedge_funds

from __future__ import annotations

import numpy as np

from hedge_fund_ml.utils.finance import (
    factor_hf_split,
    normalization,
    random_sampling,
    transaction_cost,
)


def test_normalization_matches_variance_ratio() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    beta = np.eye(2)
    scale = normalization(y, x, beta, window=2)
    expected = np.sqrt(np.var(y, axis=0, ddof=1)) / np.sqrt(
        np.var(x @ beta, axis=0, ddof=1)
    )
    np.testing.assert_allclose(scale, expected)


def test_random_sampling_is_deterministic_with_seed() -> None:
    dataset = np.arange(20).reshape(10, 2)
    rng = np.random.default_rng(42)
    sample = random_sampling(dataset, n_samples=3, window=4, rng=rng)
    indices = np.array([0, 5, 4])
    expected = np.stack([dataset[start : start + 4] for start in indices])
    np.testing.assert_array_equal(sample, expected)


def test_transaction_cost_scaling() -> None:
    old_weights = np.array([0.5, 0.5])
    new_weights = np.array([0.4, 0.3])
    covariance = np.eye(2)
    costs = transaction_cost(old_weights, new_weights, covariance, impact_scale=0.1)
    np.testing.assert_allclose(costs, np.array([0.0005, 0.002]))


def test_factor_hf_split_shapes() -> None:
    array = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    factors, hedge_funds = factor_hf_split(array, split_pos=2, reshape=True)
    assert factors.shape == (6, 2)
    assert hedge_funds.shape == (6, 2)

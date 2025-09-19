from __future__ import annotations

import numpy as np

from replicate.decoder import WeightDecoder


def test_decoder_matches_closed_form_with_turnover() -> None:
    """Synthetic ridge-style regression should match analytic solution."""

    rhat = np.array([0.12, -0.08], dtype=float)
    yhat = 0.05
    w_prev = np.array([0.03, -0.01], dtype=float)
    lambda_to = 0.2
    lambda_l2 = 0.05

    decoder = WeightDecoder(
        leverage=10.0,
        lambda_to=lambda_to,
        lambda_l2=lambda_l2,
        long_only=False,
    )
    result = decoder.solve_once(rhat, yhat, w_prev=w_prev)

    gram = np.outer(rhat, rhat)
    identity = np.eye(rhat.size)
    lhs = gram + (lambda_to + lambda_l2) * identity
    rhs = yhat * rhat + lambda_to * w_prev
    closed_form = np.linalg.solve(lhs, rhs)

    np.testing.assert_allclose(result.weights, closed_form, atol=1e-6)

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["WeightDecoder", "DecoderResult"]


@dataclass(slots=True)
class DecoderResult:
    """Container for optimization outputs."""

    weights: NDArray[np.float_]
    objective_value: float
    status: str


class WeightDecoder:
    """Solve the convex decoding problem with leverage and turnover control."""

    def __init__(
        self,
        leverage: float,
        lambda_to: float,
        lambda_l2: float,
        *,
        long_only: bool = False,
        solver: str = "OSQP",
        solver_opts: Mapping[str, Any] | None = None,
    ) -> None:
        if leverage <= 0:
            raise ValueError("leverage must be positive")
        if lambda_to < 0:
            raise ValueError("lambda_to must be non-negative")
        if lambda_l2 < 0:
            raise ValueError("lambda_l2 must be non-negative")

        self.leverage = float(leverage)
        self.lambda_to = float(lambda_to)
        self.lambda_l2 = float(lambda_l2)
        self.long_only = bool(long_only)
        self.solver = solver
        self.solver_opts: Mapping[str, Any] = solver_opts or {
            "eps_abs": 1e-6,
            "eps_rel": 1e-6,
            "max_iter": 20_000,
        }

    def solve_once(
        self,
        rhat_etf: ArrayLike,
        yhat: float,
        w_prev: ArrayLike | None = None,
    ) -> DecoderResult:
        """Decode a single period of forecasts into ETF weights."""

        rhat = np.asarray(rhat_etf, dtype=float).reshape(-1)
        if rhat.ndim != 1:
            raise ValueError("rhat_etf must be one-dimensional")
        n_assets = rhat.shape[0]
        prev = np.zeros(n_assets) if w_prev is None else np.asarray(w_prev, dtype=float).reshape(-1)
        if prev.shape != (n_assets,):
            raise ValueError("w_prev must have the same shape as rhat_etf")

        weights = cp.Variable(n_assets)
        fit = cp.sum_squares(rhat @ weights - float(yhat))
        turnover = self.lambda_to * cp.sum_squares(weights - prev)
        ridge = self.lambda_l2 * cp.sum_squares(weights)
        objective = cp.Minimize(fit + turnover + ridge)

        constraints = []
        if self.long_only:
            constraints.extend([weights >= 0, cp.sum(weights) <= self.leverage])
        else:
            constraints.append(cp.norm1(weights) <= self.leverage)

        problem = cp.Problem(objective, constraints)
        weights.value = prev.copy()
        problem.solve(solver=getattr(cp, self.solver), **self.solver_opts)

        if weights.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"Decoder failed: status={problem.status}")

        solution = np.asarray(weights.value, dtype=float)
        return DecoderResult(
            weights=solution,
            objective_value=float(problem.value),
            status=problem.status,
        )

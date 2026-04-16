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

        # Internal state for parameterized problem caching
        self._n_assets: int | None = None
        self._param_rhat: cp.Parameter | None = None
        self._param_yhat: cp.Parameter | None = None
        self._param_prev: cp.Parameter | None = None
        self._var_weights: cp.Variable | None = None
        self._problem: cp.Problem | None = None

    def _setup_problem(self, n_assets: int) -> None:
        """Compile the parameterized CVXPY problem once for significant speedups."""
        self._n_assets = n_assets
        self._param_rhat = cp.Parameter(n_assets)
        self._param_yhat = cp.Parameter()
        self._param_prev = cp.Parameter(n_assets)
        self._var_weights = cp.Variable(n_assets)

        fit = cp.sum_squares(self._param_rhat @ self._var_weights - self._param_yhat)
        turnover = self.lambda_to * cp.sum_squares(self._var_weights - self._param_prev)
        ridge = self.lambda_l2 * cp.sum_squares(self._var_weights)
        objective = cp.Minimize(fit + turnover + ridge)

        constraints = []
        if self.long_only:
            constraints.extend([self._var_weights >= 0, cp.sum(self._var_weights) <= self.leverage])
        else:
            constraints.append(cp.norm1(self._var_weights) <= self.leverage)

        self._problem = cp.Problem(objective, constraints)

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

        # ⚡ Bolt Optimization: Lazy compilation and parameterization of CVXPY problem
        # Compiling cp.Problem is extremely slow. Using cp.Parameter allows us to compile it
        # exactly once per asset dimension size and just update parameter values thereafter.
        if self._n_assets != n_assets:
            self._setup_problem(n_assets)

        self._param_rhat.value = rhat
        self._param_yhat.value = float(yhat)
        self._param_prev.value = prev
        self._var_weights.value = prev.copy()

        self._problem.solve(solver=getattr(cp, self.solver), **self.solver_opts)

        valid_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        if self._var_weights.value is None or self._problem.status not in valid_statuses:
            raise RuntimeError(f"Decoder failed: status={self._problem.status}")

        solution = np.asarray(self._var_weights.value, dtype=float)
        return DecoderResult(
            weights=solution,
            objective_value=float(self._problem.value),
            status=self._problem.status,
        )

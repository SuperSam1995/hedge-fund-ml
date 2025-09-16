"""Tests for the HK span regression helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import HKSpanConfig, HKSpanModel


def test_hk_span_train_then_predict(tmp_path: Path) -> None:
    index = pd.date_range("2010-01-31", periods=12, freq="ME")
    span = pd.DataFrame(
        {
            "f1": np.linspace(-0.05, 0.08, len(index)),
            "f2": np.linspace(0.02, -0.04, len(index)),
        },
        index=index,
    )
    beta = np.array([[0.3, -0.2], [0.5, 0.4]])
    intercept = np.array([0.01, -0.015])
    target = pd.DataFrame(
        span.to_numpy() @ beta + intercept,
        index=index,
        columns=["y1", "y2"],
    )

    train_span = span.iloc[:8]
    train_target = target.iloc[:8]
    test_span = span.iloc[8:]
    test_target = target.iloc[8:]

    model = HKSpanModel(HKSpanConfig(add_intercept=True))
    model.fit(train_span, train_target)
    assert model.state is not None
    state = model.state

    assert state.coefficients.shape == beta.shape
    assert state.intercept.shape == (beta.shape[1],)

    predictions = model.predict(test_span)
    expected = pd.DataFrame(
        test_span.to_numpy() @ beta + intercept,
        index=test_span.index,
        columns=test_target.columns,
    )
    np.testing.assert_allclose(predictions.to_numpy(), expected.to_numpy())

    residuals = model.residuals(test_target, test_span)
    np.testing.assert_allclose(
        residuals.to_numpy(), (test_target - expected).to_numpy(), atol=1e-12
    )

    # Persist and reload the model to ensure coefficients survive round-trips.
    model_path = tmp_path / "hk_span.json"
    model.dump(model_path)
    loaded = HKSpanModel.load(model_path)
    reloaded_predictions = loaded.predict(test_span)
    np.testing.assert_allclose(reloaded_predictions.to_numpy(), expected.to_numpy())

    # Changing the test targets should only affect residuals, not predictions.
    bump = pd.DataFrame(
        np.zeros_like(test_target.to_numpy()),
        index=test_target.index,
        columns=test_target.columns,
    )
    bump.iloc[0, 0] = 1.0
    shifted = test_target + bump
    shifted_residuals = loaded.residuals(shifted, test_span)
    np.testing.assert_allclose(
        shifted_residuals.to_numpy(), (shifted - expected).to_numpy(), atol=1e-12
    )

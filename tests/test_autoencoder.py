from __future__ import annotations

import numpy as np
import pandas as pd

from hedge_fund_ml.models import (
    AutoencoderConfig,
    AutoencoderModelConfig,
    AutoencoderOutputConfig,
    AutoencoderTrainingConfig,
    build_model,
    fit,
    transform,
)


def _make_config(tmp_path) -> AutoencoderConfig:
    return AutoencoderConfig(
        seed=0,
        model=AutoencoderModelConfig(
            input_dim=6,
            latent_dim=3,
            hidden_dims=[8],
            activation="relu",
            l2=1e-5,
        ),
        training=AutoencoderTrainingConfig(
            epochs=2,
            batch_size=16,
            patience=1,
            verbose=0,
        ),
        output=AutoencoderOutputConfig(root=tmp_path),
    )


def test_build_model_round_trip(tmp_path) -> None:
    cfg = _make_config(tmp_path)
    model = build_model(cfg)
    assert model.input_shape == (None, 6)
    assert model.output_shape == (None, 6)


def test_transform_preserves_row_count(tmp_path) -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        rng.normal(size=(64, 6)).astype(np.float32),
        columns=[f"f{i}" for i in range(6)],
    )
    cfg = _make_config(tmp_path)
    train = data.iloc[:48]
    val = data.iloc[48:]

    result = fit(train, val, cfg)
    transform_cfg = cfg.model_copy(
        update={
            "output": AutoencoderOutputConfig(
                root=cfg.output.root, run_path=result.run_dir
            )
        }
    )
    latent = transform(data, transform_cfg)
    assert latent.shape[0] == data.shape[0]
    assert latent.shape[1] == cfg.model.latent_dim

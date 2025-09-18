import numpy as np

from hedge_fund_ml.models import (
    WGANConfig,
    WGANModelConfig,
    WGANOutputConfig,
    WGANTrainingConfig,
    sample,
    train_gan,
)


def _make_config(tmp_path) -> WGANConfig:
    return WGANConfig(
        seed=0,
        model=WGANModelConfig(
            latent_dim=4,
            generator_units=[8],
            critic_units=[8],
            activation="relu",
            output_activation="tanh",
            critic_slope=0.2,
        ),
        training=WGANTrainingConfig(
            epochs=2,
            batch_size=8,
            n_critic=1,
            clip_value=0.05,
            learning_rate=1e-3,
            patience=1,
            min_delta=1e-3,
        ),
        output=WGANOutputConfig(root=tmp_path / "wgan"),
    )


def test_train_and_sample_shape(tmp_path) -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(32, 4, 3)).astype(np.float32)
    cfg = _make_config(tmp_path)

    artifacts = train_gan(data, cfg)
    assert artifacts.run_dir.exists()
    assert not artifacts.history.empty

    sample_cfg = cfg.model_copy(
        update={"output": WGANOutputConfig(root=cfg.output.root, run_path=artifacts.run_dir)}
    )
    generated = sample(5, sample_cfg)
    assert generated.shape == (5, 4, 3)

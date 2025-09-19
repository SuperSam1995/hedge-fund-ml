from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_transformer_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train_transformer.py"
    spec = importlib.util.spec_from_file_location("train_transformer", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to load train_transformer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transformer_module_single_batch_forward_backward() -> None:
    torch.manual_seed(0)
    module = _load_transformer_module()

    model_cfg = module.TransformerModelConfig(d_model=8, n_heads=2, depth=1, dropout=0.0, attn_bias=False)
    train_cfg = module.TrainingConfig(
        batch_size=4,
        max_epochs=1,
        early_stopping_patience=1,
        lr=1e-3,
        weight_decay=0.0,
        seed=0,
    )

    transformer = module.TransformerModule(
        model_config=model_cfg,
        train_config=train_cfg,
        input_dim=3,
        seq_len=5,
        target_dim=1,
    )

    features = torch.randn(train_cfg.batch_size, 5, 3)
    targets = torch.randn(train_cfg.batch_size, 1)

    loss = transformer.training_step((features, targets), batch_idx=0)
    assert torch.isfinite(loss)

    loss.backward()
    grads = [param.grad for param in transformer.parameters() if param.requires_grad]
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads)

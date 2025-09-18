"""Reproducible experiment helpers."""

from __future__ import annotations

import importlib
import importlib.util
import os
import random

import numpy as np

__all__ = ["set_global_seed"]


def _optional_import(name: str):
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


def set_global_seed(seed: int) -> dict[str, str]:
    """Seed Python, NumPy, PyTorch and TensorFlow if available."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    affected: dict[str, str] = {"python": str(seed), "numpy": str(seed)}

    torch = _optional_import("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        affected["torch"] = str(seed)

    tensorflow = _optional_import("tensorflow")
    if tensorflow is not None:
        tensorflow.random.set_seed(seed)
        affected["tensorflow"] = str(seed)

    return affected

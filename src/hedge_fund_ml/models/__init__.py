"""Model implementations."""

from __future__ import annotations

from .autoencoder import (
    AutoencoderArtifacts,
    AutoencoderConfig,
    AutoencoderDataConfig,
    AutoencoderModelConfig,
    AutoencoderOutputConfig,
    AutoencoderTrainingConfig,
    build_model,
    fit,
    transform,
)
from .gan import (
    WGANArtifacts,
    WGANConfig,
    WGANModelConfig,
    WGANOutputConfig,
    WGANTrainingConfig,
    build_gan,
    sample,
    train_gan,
)

__all__ = [
    "AutoencoderArtifacts",
    "AutoencoderConfig",
    "AutoencoderDataConfig",
    "AutoencoderModelConfig",
    "AutoencoderOutputConfig",
    "AutoencoderTrainingConfig",
    "WGANArtifacts",
    "WGANConfig",
    "WGANModelConfig",
    "WGANOutputConfig",
    "WGANTrainingConfig",
    "build_gan",
    "build_model",
    "fit",
    "sample",
    "train_gan",
    "transform",
]

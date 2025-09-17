"""GAN-related utilities."""

from __future__ import annotations

from .wgan import (
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
    "WGANArtifacts",
    "WGANConfig",
    "WGANModelConfig",
    "WGANOutputConfig",
    "WGANTrainingConfig",
    "build_gan",
    "sample",
    "train_gan",
]

"""Inference utilities for road generation."""

from .blending import (
    SeamlessGenerator,
    blend_patches,
    compute_patch_grid,
    create_blend_weights,
    PatchInfo,
)

from .reconstruction import (
    reconstruct_image,
)
from .noise_strategies import (
    NoiseStrategy,
    GaussianNoiseStrategy,
    SparseNoiseStrategy,
    UniformNoiseStrategy,
    create_noise_strategy,
    is_empty_patch,
)

__all__ = [
    # Blending
    "SeamlessGenerator",
    "blend_patches",
    "compute_patch_grid",
    "create_blend_weights",
    "PatchInfo",

    # Reconstruction
    "reconstruct_image",
    # Noise strategies
    "NoiseStrategy",
    "GaussianNoiseStrategy",
    "SparseNoiseStrategy",
    "UniformNoiseStrategy",
    "create_noise_strategy",
    "is_empty_patch",
]
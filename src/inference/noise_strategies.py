"""
Noise strategies for handling empty patches in image generation.

This module provides various noise injection strategies to handle empty or
uniform patches during seamless image generation, preventing artifacts from
feeding empty inputs to the model.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class NoiseStrategy(ABC):
    """Base class for noise injection strategies."""

    @abstractmethod
    def add_noise(self, patch: np.ndarray, is_empty: bool = False) -> np.ndarray:
        """
        Add noise to a patch.

        Args:
            patch: Input patch (H, W, C) in range [0, 255]
            is_empty: Whether the patch is considered empty

        Returns:
            Patch with noise added (H, W, C) in range [0, 255]
        """
        pass


class NoNoiseStrategy(NoiseStrategy):
    """Strategy that adds no noise."""

    def add_noise(self, patch: np.ndarray, is_empty: bool = False) -> np.ndarray:
        return patch.copy()


class GaussianNoiseStrategy(NoiseStrategy):
    """Add Gaussian noise to patches."""

    def __init__(self, mean: float = 0.0, std: float = 10.0, only_if_empty: bool = True):
        """
        Initialize Gaussian noise strategy.

        Args:
            mean: Mean of the Gaussian noise
            std: Standard deviation of the Gaussian noise
            only_if_empty: Only add noise to empty patches
        """
        self.mean = mean
        self.std = std
        self.only_if_empty = only_if_empty

    def add_noise(self, patch: np.ndarray, is_empty: bool = False) -> np.ndarray:
        if self.only_if_empty and not is_empty:
            return patch.copy()

        noise = np.random.normal(self.mean, self.std, patch.shape)
        noisy_patch = patch.astype(np.float32) + noise
        return np.clip(noisy_patch, 0, 255).astype(np.uint8)


class UniformNoiseStrategy(NoiseStrategy):
    """Add uniform noise to patches."""

    def __init__(self, low: float = -10.0, high: float = 10.0, only_if_empty: bool = True):
        """
        Initialize uniform noise strategy.

        Args:
            low: Lower bound of uniform noise
            high: Upper bound of uniform noise
            only_if_empty: Only add noise to empty patches
        """
        self.low = low
        self.high = high
        self.only_if_empty = only_if_empty

    def add_noise(self, patch: np.ndarray, is_empty: bool = False) -> np.ndarray:
        if self.only_if_empty and not is_empty:
            return patch.copy()

        noise = np.random.uniform(self.low, self.high, patch.shape)
        noisy_patch = patch.astype(np.float32) + noise
        return np.clip(noisy_patch, 0, 255).astype(np.uint8)


class SparseNoiseStrategy(NoiseStrategy):
    """Add sparse noise to patches (only to a subset of pixels)."""

    def __init__(self, density: float = 0.1, magnitude: float = 20.0, only_if_empty: bool = True):
        """
        Initialize sparse noise strategy.

        Args:
            density: Fraction of pixels to add noise to (0-1)
            magnitude: Magnitude of the noise
            only_if_empty: Only add noise to empty patches
        """
        self.density = np.clip(density, 0.0, 1.0)
        self.magnitude = magnitude
        self.only_if_empty = only_if_empty

    def add_noise(self, patch: np.ndarray, is_empty: bool = False) -> np.ndarray:
        if self.only_if_empty and not is_empty:
            return patch.copy()

        noisy_patch = patch.astype(np.float32).copy()

        # Create mask for pixels to add noise to
        mask = np.random.random(patch.shape[:2]) < self.density

        # Add noise to selected pixels
        noise = np.random.normal(0, self.magnitude, patch.shape)
        noisy_patch[mask] += noise[mask]

        return np.clip(noisy_patch, 0, 255).astype(np.uint8)


class PerChannelNoiseStrategy(NoiseStrategy):
    """Add different noise to each channel independently."""

    def __init__(
        self,
        mean: float = 0.0,
        std_range: tuple = (5.0, 15.0),
        only_if_empty: bool = True
    ):
        """
        Initialize per-channel noise strategy.

        Args:
            mean: Mean of the Gaussian noise
            std_range: Range (min, max) for randomly sampling std per channel
            only_if_empty: Only add noise to empty patches
        """
        self.mean = mean
        self.std_range = std_range
        self.only_if_empty = only_if_empty

    def add_noise(self, patch: np.ndarray, is_empty: bool = False) -> np.ndarray:
        if self.only_if_empty and not is_empty:
            return patch.copy()

        noisy_patch = patch.astype(np.float32).copy()

        # Add different noise to each channel
        for c in range(patch.shape[2]):
            std = np.random.uniform(*self.std_range)
            noise = np.random.normal(self.mean, std, patch.shape[:2])
            noisy_patch[:, :, c] += noise

        return np.clip(noisy_patch, 0, 255).astype(np.uint8)


def create_noise_strategy(
    strategy: str,
    mean: float = 0.0,
    std: float = 10.0,
    low: float = -10.0,
    high: float = 10.0,
    density: float = 0.1,
    magnitude: float = 20.0,
    std_range: tuple = (5.0, 15.0),
    only_if_empty: bool = True,
    **kwargs
) -> NoiseStrategy:
    """
    Factory function to create noise strategies.

    Args:
        strategy: Name of the strategy ("gaussian", "uniform", "sparse", "per_channel", "none")
        mean: Mean for Gaussian noise strategies
        std: Standard deviation for Gaussian noise
        low: Lower bound for uniform noise
        high: Upper bound for uniform noise
        density: Density for sparse noise
        magnitude: Magnitude for sparse noise
        std_range: Range for per-channel noise
        only_if_empty: Only add noise to empty patches
        **kwargs: Additional arguments (ignored)

    Returns:
        Configured NoiseStrategy instance
    """
    strategy = strategy.lower()

    if strategy == "none":
        return NoNoiseStrategy()
    elif strategy == "gaussian":
        return GaussianNoiseStrategy(mean=mean, std=std, only_if_empty=only_if_empty)
    elif strategy == "uniform":
        return UniformNoiseStrategy(low=low, high=high, only_if_empty=only_if_empty)
    elif strategy == "sparse":
        return SparseNoiseStrategy(
            density=density,
            magnitude=magnitude,
            only_if_empty=only_if_empty
        )
    elif strategy == "per_channel":
        return PerChannelNoiseStrategy(
            mean=mean,
            std_range=std_range,
            only_if_empty=only_if_empty
        )
    else:
        raise ValueError(
            f"Unknown noise strategy: {strategy}. "
            f"Available strategies: gaussian, uniform, sparse, per_channel, none"
        )


def is_empty_patch(patch: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Check if a patch is mostly empty or uniform.

    A patch is considered empty if it has very low variance, indicating
    it's mostly a single color or very uniform.

    Args:
        patch: Input patch (H, W, C) in range [0, 255]
        threshold: Variance threshold (normalized to [0, 1])

    Returns:
        True if the patch is considered empty
    """
    # Normalize to [0, 1] and compute variance
    normalized = patch.astype(np.float32) / 255.0
    variance = np.var(normalized)

    return variance < threshold

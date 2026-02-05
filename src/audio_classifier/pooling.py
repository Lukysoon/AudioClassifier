"""
Pooling strategies for aggregating temporal HuBERT features.
"""

import torch
from abc import ABC, abstractmethod
from typing import Literal


class PoolingStrategy(ABC):
    """Abstract base class for pooling strategies."""

    @abstractmethod
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden states over the temporal dimension.

        Args:
            hidden_states: Tensor of shape [batch, seq_len, dim].

        Returns:
            Pooled tensor of shape [batch, dim].
        """
        pass


class MeanPooling(PoolingStrategy):
    """Average pooling over temporal dimension."""

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute mean over sequence length.

        This captures the average representation across all time frames,
        giving equal weight to all parts of the audio.
        """
        return hidden_states.mean(dim=1)


class MaxPooling(PoolingStrategy):
    """Max pooling over temporal dimension."""

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Take maximum value for each feature dimension.

        This captures the most prominent activations,
        useful for detecting specific audio events.
        """
        return hidden_states.max(dim=1).values


def get_pooling_strategy(name: Literal["mean", "max"]) -> PoolingStrategy:
    """
    Factory function for pooling strategies.

    Args:
        name: Name of the pooling strategy.

    Returns:
        Instance of the requested pooling strategy.

    Raises:
        ValueError: If unknown pooling strategy name.
    """
    strategies = {
        "mean": MeanPooling,
        "max": MaxPooling,
    }

    if name not in strategies:
        available = list(strategies.keys())
        raise ValueError(f"Unknown pooling strategy: '{name}'. Available: {available}")

    return strategies[name]()

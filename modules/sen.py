# Module description: lightweight temporal squeeze-and-excitation for gloss sequence features.

from __future__ import annotations

import torch
from torch import nn


class TemporalSEN(nn.Module):
    """Channel reweighting block for temporal sequence features.

    This module expects sequence features with shape ``(batch, time, channels)`` and
    applies squeeze-and-excitation across the temporal axis:

    1. Average-pool over the ``time`` dimension to get a channel descriptor
       with shape ``(batch, channels)``.
    2. Pass the descriptor through a two-layer bottleneck MLP
       ``channels -> channels // reduction -> channels``.
    3. Apply a sigmoid gate and rescale the original sequence features.

    Args:
        channels: Feature channel size ``C`` of the input tensor.
        reduction: Reduction ratio for the bottleneck hidden dimension.

    Shape:
        - Input: ``(B, T, C)``
        - Output: ``(B, T, C)``
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")

        reduced_channels = max(1, channels // reduction)
        self.channels = channels
        self.reduction = reduction
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the two-layer bottleneck with Xavier-friendly defaults."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal squeeze-and-excitation.

        Args:
            x: Sequence features with shape ``(B, T, C)``.

        Returns:
            Tensor with the same shape ``(B, T, C)`` after channel-wise rescaling.
        """
        if x.dim() != 3:
            raise ValueError("TemporalSEN expects input with shape (batch, time, channels)")
        if x.size(-1) != self.channels:
            raise ValueError(f"Expected last dimension {self.channels}, got {x.size(-1)}")

        # (B, T, C) -> (B, C)
        pooled = x.mean(dim=1)
        # (B, C) -> (B, C // reduction) -> (B, C)
        weights = self.fc2(self.activation(self.fc1(pooled)))
        # (B, C) -> (B, 1, C)
        weights = self.gate(weights).unsqueeze(1)
        # Broadcast weights along the time dimension: (B, T, C)
        return x * weights


if __name__ == "__main__":
    torch.manual_seed(7)
    module = TemporalSEN(channels=128, reduction=16)
    dummy = torch.randn(2, 12, 128)
    output = module(dummy)
    print(f"input shape: {tuple(dummy.shape)}")
    print(f"output shape: {tuple(output.shape)}")

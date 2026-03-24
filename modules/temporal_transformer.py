# Module description: lightweight temporal transformer encoder for gloss sequence modeling.

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for ``(B, T, C)`` sequence features."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            encoding[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to the input sequence tensor."""
        seq_len = x.size(1)
        if seq_len > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds supported maximum {self.encoding.size(1)}. "
                "Increase max_len when constructing SinusoidalPositionEncoding."
            )
        return x + self.encoding[:, :seq_len].to(dtype=x.dtype, device=x.device)


class MultiHeadSelfAttention(nn.Module):
    """ONNX-friendly multi-head self-attention for temporal features."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize attention projections."""
        for layer in (self.query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        # (B, T, C) -> (B, H, T, C // H)
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute self-attention on ``(B, T, C)`` features.

        Args:
            x: Input tensor with shape ``(B, T, C)``.
            padding_mask: Optional validity mask with shape ``(B, T)`` where
                ``True`` means a valid token and ``False`` means padding.

        Returns:
            Tensor with shape ``(B, T, C)``.
        """
        # (B, T, C) -> (B, H, T, D)
        query = self._reshape_heads(self.query(x))
        key = self._reshape_heads(self.key(x))
        value = self._reshape_heads(self.value(x))

        # Attention logits: (B, H, T, D) x (B, H, D, T) -> (B, H, T, T)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            attn_mask = padding_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~attn_mask, -1e4)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # (B, H, T, T) x (B, H, T, D) -> (B, H, T, D)
        attended = torch.matmul(weights, value)
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        attended = attended.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        output = self.output(attended)

        if padding_mask is not None:
            output = output.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        return output


class TransformerEncoderLayer(nn.Module):
    """Single pre-layer-normalized transformer encoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize feed-forward layers while leaving layer norms at defaults."""
        linear_layers = [module for module in self.ffn if isinstance(module, nn.Linear)]
        for layer in linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply pre-LN self-attention and feed-forward sublayers.

        Args:
            x: Input tensor with shape ``(B, T, C)``.
            padding_mask: Optional validity mask with shape ``(B, T)``.

        Returns:
            Tensor with shape ``(B, T, C)``.
        """
        attn_input = self.norm1(x)
        x = x + self.dropout(self.self_attn(attn_input, padding_mask=padding_mask))

        ffn_input = self.norm2(x)
        x = x + self.dropout(self.ffn(ffn_input))

        if padding_mask is not None:
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        return x


class TemporalTransformerEncoder(nn.Module):
    """Lightweight transformer encoder for gloss temporal modeling.

    The module keeps the input and output shapes identical so it can be inserted
    before the existing BiGRU encoder without changing downstream decoder or
    attention interfaces.

    Args:
        d_model: Feature dimension ``C`` of the input tensor.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        dropout: Dropout rate used in attention, feed-forward layers, and after
            positional encoding.
        d_ff: Hidden size of the feed-forward network. Defaults to ``4 * d_model``.
        max_len: Maximum supported sequence length for positional encoding.

    Shape:
        - Input: ``(B, T, C)``
        - Padding mask: ``(B, T)`` with ``True`` for valid positions
        - Output: ``(B, T, C)``
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        d_ff: Optional[int] = None,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.d_model = d_model
        self.position_encoding = SinusoidalPositionEncoding(d_model=d_model, max_len=max_len)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff or (4 * d_model),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode temporal gloss embeddings.

        Args:
            x: Input features with shape ``(B, T, C)``.
            padding_mask: Optional validity mask with shape ``(B, T)``.

        Returns:
            Tensor with shape ``(B, T, C)``.
        """
        if x.dim() != 3:
            raise ValueError("TemporalTransformerEncoder expects input with shape (batch, time, channels)")

        output = self.position_encoding(x)
        output = self.input_dropout(output)
        if padding_mask is not None:
            output = output.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        for layer in self.layers:
            output = layer(output, padding_mask=padding_mask)

        output = self.final_norm(output)
        if padding_mask is not None:
            output = output.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        return output


if __name__ == "__main__":
    torch.manual_seed(7)
    module = TemporalTransformerEncoder(d_model=128, num_layers=2, num_heads=4, dropout=0.1)
    dummy = torch.randn(2, 12, 128)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, False, False, False, False, False, False],
            [True] * 10 + [False, False],
        ]
    )
    output = module(dummy, padding_mask=mask)
    print(f"input shape: {tuple(dummy.shape)}")
    print(f"mask shape: {tuple(mask.shape)}")
    print(f"output shape: {tuple(output.shape)}")

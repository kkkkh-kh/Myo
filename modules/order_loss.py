# Module description: Auxiliary losses for improving gloss-to-Chinese word-order learning.

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class WordOrderLoss(nn.Module):
    """Combine CE loss with attention-based word-order auxiliary losses."""

    def __init__(
        self,
        alpha_mono: float = 0.1,
        alpha_order: float = 0.05,
        warmup_epochs: int = 10,
        order_interval: int = 10,
    ) -> None:
        """Initialize loss weights.

        Args:
            alpha_mono: Weight for monotonicity loss.
            alpha_order: Weight for order-consistency loss.
            warmup_epochs: Linear warmup epochs for auxiliary weights.
            order_interval: Compute order-consistency every N steps.
        """
        super().__init__()
        self.alpha_mono = float(alpha_mono)
        self.alpha_order = float(alpha_order)
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.order_interval = max(1, int(order_interval))

    def attention_monotonicity_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Penalize backward movement of attention expectation.

        Args:
            attention_weights: Tensor with shape ``(B, T_dec, T_enc)``.

        Returns:
            Scalar monotonicity loss.
        """
        if attention_weights.ndim != 3 or attention_weights.size(1) <= 1:
            return attention_weights.sum() * 0.0

        # (T_enc,) -> (1, 1, T_enc)
        positions = torch.arange(attention_weights.size(-1), device=attention_weights.device, dtype=attention_weights.dtype)
        positions = positions.view(1, 1, -1)
        # (B, T_dec, T_enc) * (1, 1, T_enc) -> (B, T_dec)
        expected_positions = (attention_weights * positions).sum(dim=-1)

        # Penalize expected_pos(i-1) > expected_pos(i)
        regressions = F.relu(expected_positions[:, :-1] - expected_positions[:, 1:])
        return regressions.mean()

    def order_consistency_loss(
        self,
        attention_weights: torch.Tensor,
        order_patterns: List[str],
    ) -> torch.Tensor:
        """Encourage similar attention maps for samples with same order pattern.

        Args:
            attention_weights: Tensor with shape ``(B, T_dec, T_enc)``.
            order_patterns: Pattern labels for each sample in the batch.

        Returns:
            Scalar order-consistency loss.
        """
        if attention_weights.ndim != 3 or attention_weights.size(0) < 2:
            return attention_weights.sum() * 0.0
        if order_patterns is None or len(order_patterns) != attention_weights.size(0):
            return attention_weights.sum() * 0.0

        flattened = attention_weights.reshape(attention_weights.size(0), -1)
        flattened = F.normalize(flattened, dim=-1)

        similarities = []
        for left in range(len(order_patterns)):
            for right in range(left + 1, len(order_patterns)):
                if order_patterns[left] != order_patterns[right]:
                    continue
                similarity = (flattened[left] * flattened[right]).sum()
                similarities.append(similarity)

        if not similarities:
            return attention_weights.sum() * 0.0
        mean_similarity = torch.stack(similarities).mean()
        return 1.0 - mean_similarity

    def _warmup_scale(self, current_epoch: int) -> float:
        progress = float(current_epoch + 1) / float(self.warmup_epochs)
        return max(0.0, min(1.0, progress))

    def forward(
        self,
        ce_loss: torch.Tensor,
        attention_weights: torch.Tensor,
        order_patterns: List[str] = None,
        current_epoch: int = 0,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss and a logging-friendly breakdown.

        Args:
            ce_loss: Base CE or label-smoothed CE loss.
            attention_weights: Tensor with shape ``(B, T_dec, T_enc)``.
            order_patterns: Optional order labels for consistency loss.
            current_epoch: Current epoch index.
            step_idx: Current optimization step index within epoch.

        Returns:
            ``(total_loss, breakdown_dict)``.
        """
        if attention_weights is None or attention_weights.numel() == 0:
            breakdown = {"ce": float(ce_loss.detach().item()), "mono": 0.0, "order": 0.0}
            return ce_loss, breakdown

        mono_loss = self.attention_monotonicity_loss(attention_weights)
        order_loss = attention_weights.sum() * 0.0
        if order_patterns is not None and step_idx % self.order_interval == 0:
            order_loss = self.order_consistency_loss(attention_weights, order_patterns)

        warmup = self._warmup_scale(current_epoch)
        mono_term = warmup * self.alpha_mono * mono_loss
        order_term = warmup * self.alpha_order * order_loss
        total_loss = ce_loss + mono_term + order_term

        breakdown = {
            "ce": float(ce_loss.detach().item()),
            "mono": float(mono_term.detach().item()),
            "order": float(order_term.detach().item()),
        }
        return total_loss, breakdown


if __name__ == "__main__":
    loss_fn = WordOrderLoss()
    ce = torch.tensor(2.5)
    attention = torch.softmax(torch.randn(4, 10, 20), dim=-1)
    total, components = loss_fn(ce, attention, current_epoch=5)
    print(f"Total: {total.item():.4f}, Breakdown: {components}")

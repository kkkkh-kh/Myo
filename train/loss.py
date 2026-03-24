import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """KL-divergence label smoothing with PAD masking."""

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError("logits must have shape [batch, seq_len, vocab_size]")
        if target.dim() != 2:
            raise ValueError("target must have shape [batch, seq_len]")

        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        valid_mask = target.ne(self.ignore_index)
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        target = target[valid_mask]
        log_probs = F.log_softmax(logits, dim=-1)

        denominator = self.vocab_size - 2 if 0 <= self.ignore_index < self.vocab_size else self.vocab_size - 1
        smoothing_value = self.smoothing / max(1, denominator)
        true_dist = torch.full_like(log_probs, smoothing_value)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        if 0 <= self.ignore_index < self.vocab_size:
            true_dist[:, self.ignore_index] = 0.0
            normalization = true_dist.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            true_dist = true_dist / normalization

        loss = F.kl_div(log_probs, true_dist, reduction="batchmean")
        return loss

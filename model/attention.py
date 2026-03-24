from typing import Optional, Tuple

import torch
from torch import nn


class BahdanauAttention(nn.Module):
    """Additive attention with masking support."""

    def __init__(self, enc_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.w1 = nn.Linear(enc_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(
        self,
        enc_output: torch.Tensor,
        dec_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = enc_output.abs().sum(dim=-1).ne(0)
        mask = mask.to(dtype=torch.bool)

        projected_encoder = self.w1(enc_output)
        projected_hidden = self.w2(dec_hidden).unsqueeze(1)
        scores = self.v(torch.tanh(projected_encoder + projected_hidden)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)

        max_scores = scores.max(dim=-1, keepdim=True).values
        stable_scores = torch.exp(scores - max_scores)
        stable_scores = stable_scores * mask.to(stable_scores.dtype)
        denominator = stable_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attn_weights = stable_scores / denominator

        context = torch.bmm(attn_weights.unsqueeze(1), enc_output).squeeze(1)
        return context, attn_weights

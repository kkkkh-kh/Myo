# Module description: Word-order-aware attention with relative-position bias and soft guidance prior.

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class WordOrderAwareAttention(nn.Module):
    """Additive attention enhanced by relative-position bias."""

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_size: int,
        max_relative_position: int = 64,
        use_order_guidance: bool = True,
        guidance_lambda_init: float = 1.0,
        guidance_decay_ratio: float = 0.3,
    ) -> None:
        """Initialize attention projections and position priors.

        Args:
            encoder_hidden_size: Encoder feature size.
            decoder_hidden_size: Decoder hidden size.
            attention_size: Internal attention MLP size.
            max_relative_position: Relative position clipping radius.
            use_order_guidance: Whether to enable training-time soft guidance.
            guidance_lambda_init: Initial prior strength.
            guidance_decay_ratio: Epoch ratio used for linear decay.
        """
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_hidden_size, attention_size, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden_size, attention_size, bias=False)
        self.position_projection = nn.Linear(1, attention_size, bias=False)
        self.energy_projection = nn.Linear(attention_size, 1, bias=False)

        self.max_relative_position = max(1, int(max_relative_position))
        self.relative_bias = nn.Embedding(self.max_relative_position * 2 + 1, 1)
        self.use_order_guidance = bool(use_order_guidance)
        self.guidance_lambda_init = float(guidance_lambda_init)
        self.guidance_decay_ratio = max(0.0, float(guidance_decay_ratio))
        self.register_buffer("_guidance_lambda", torch.tensor(self.guidance_lambda_init), persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.encoder_projection.weight)
        nn.init.xavier_uniform_(self.decoder_projection.weight)
        nn.init.xavier_uniform_(self.position_projection.weight)
        nn.init.xavier_uniform_(self.energy_projection.weight)
        nn.init.zeros_(self.relative_bias.weight)

    def update_guidance_lambda(self, current_epoch: int, total_epochs: int) -> None:
        """Update guidance strength using a front-loaded linear decay schedule."""
        if not self.use_order_guidance:
            self._guidance_lambda.fill_(0.0)
            return
        decay_epochs = max(1, int(round(max(1, total_epochs) * self.guidance_decay_ratio)))
        if current_epoch >= decay_epochs:
            self._guidance_lambda.fill_(0.0)
            return
        ratio = max(0.0, 1.0 - float(current_epoch) / float(decay_epochs))
        self._guidance_lambda.fill_(self.guidance_lambda_init * ratio)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        current_step: int = 0,
        total_steps: int = 1,
        current_epoch: int = 0,
        total_epochs: int = 80,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute context and attention weights.

        Args:
            encoder_outputs: Tensor with shape ``(B, T_enc, H_enc)``.
            decoder_hidden: Tensor with shape ``(B, H_dec)``.
            current_step: Current decoder step ``i``.
            total_steps: Target length ``T_dec``.
            current_epoch: Current training epoch.
            total_epochs: Total training epochs.
            mask: Optional source padding mask with shape ``(B, T_enc)``.

        Returns:
            ``(context_vector, attention_weights)`` with shapes ``(B, H_enc)`` and ``(B, T_enc)``.
        """
        if mask is None:
            mask = encoder_outputs.abs().sum(dim=-1).ne(0)
        mask = mask.to(dtype=torch.bool)
        _, src_len, _ = encoder_outputs.shape

        if self.training:
            self.update_guidance_lambda(current_epoch=current_epoch, total_epochs=total_epochs)
        else:
            self._guidance_lambda.fill_(0.0)
        device = encoder_outputs.device

        # (B, T_enc, H_enc) -> (B, T_enc, A)
        projected_encoder = self.encoder_projection(encoder_outputs)
        # (B, H_dec) -> (B, 1, A)
        projected_decoder = self.decoder_projection(decoder_hidden).unsqueeze(1)

        src_positions = torch.arange(src_len, device=device)
        relative = int(current_step) - src_positions
        relative = torch.clamp(relative, min=-self.max_relative_position, max=self.max_relative_position)
        relative_ids = relative + self.max_relative_position

        # (T_enc,) -> (1, T_enc, 1) -> (1, T_enc, A)
        relative_scalars = self.relative_bias(relative_ids).view(1, src_len, 1)
        projected_position = self.position_projection(relative_scalars)

        # (B, T_enc, A) + (B, 1, A) + (1, T_enc, A) -> (B, T_enc)
        energy = self.energy_projection(torch.tanh(projected_encoder + projected_decoder + projected_position)).squeeze(-1)

        if self.use_order_guidance and self._guidance_lambda.item() > 0:
            target_center = float(current_step) * (float(src_len) / float(max(1, total_steps)))
            distance = (src_positions.to(dtype=encoder_outputs.dtype) - target_center) / max(1.0, float(src_len))
            prior = -self._guidance_lambda * (distance**2)
            energy = energy + prior.unsqueeze(0)

        energy = energy.masked_fill(~mask, -1e9)
        energy_max = energy.max(dim=-1, keepdim=True).values
        stable = torch.exp(energy - energy_max) * mask.to(dtype=encoder_outputs.dtype)
        denominator = stable.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attention_weights = stable / denominator

        # (B, 1, T_enc) x (B, T_enc, H_enc) -> (B, H_enc)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attention_weights


if __name__ == "__main__":
    attention = WordOrderAwareAttention(
        encoder_hidden_size=512,
        decoder_hidden_size=512,
        attention_size=256,
    )
    encoder_output = torch.randn(4, 20, 512)
    decoder_state = torch.randn(4, 512)
    context, weights = attention(encoder_output, decoder_state, current_step=3, total_steps=10)
    assert context.shape == (4, 512)
    assert weights.shape == (4, 20)
    assert abs(weights.sum(dim=-1).mean().item() - 1.0) < 1e-5
    print("WordOrderAwareAttention OK")

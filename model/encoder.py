from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.sen import TemporalSEN
from modules.temporal_transformer import TemporalTransformerEncoder


class GlossEncoder(nn.Module):
    """Bidirectional GRU encoder with optional temporal enhancement modules."""

    def __init__(
        self,
        gloss_vocab_size: int = 4000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_id: int = 0,
        use_sen: bool = False,
        sen_reduction: int = 16,
        use_transformer: bool = False,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_id = pad_id
        self.use_sen = use_sen
        self.use_transformer = use_transformer

        self.embedding = nn.Embedding(gloss_vocab_size, embed_dim, padding_idx=pad_id)
        self.temporal_sen: Optional[TemporalSEN]
        self.temporal_transformer: Optional[TemporalTransformerEncoder]
        self.temporal_sen = TemporalSEN(embed_dim, reduction=sen_reduction) if use_sen else None
        self.temporal_transformer = (
            TemporalTransformerEncoder(
                d_model=embed_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                dropout=transformer_dropout,
                max_len=512,
            )
            if use_transformer
            else None
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize learnable parameters with Xavier-friendly defaults."""
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.hidden_projection.weight)
        nn.init.zeros_(self.hidden_projection.bias)

    def _apply_temporal_modules(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply optional SEN and transformer blocks on ``(B, T, C)`` embeddings."""
        features = embedded.masked_fill(~mask.unsqueeze(-1), 0.0)
        if self.temporal_sen is not None:
            features = self.temporal_sen(features)
            features = features.masked_fill(~mask.unsqueeze(-1), 0.0)
        if self.temporal_transformer is not None:
            features = self.temporal_transformer(features, padding_mask=mask)
        return features

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode gloss ids into contextual states and a decoder-ready summary.

        Args:
            input_ids: Gloss token ids with shape ``(B, T)``.

        Returns:
            A tuple ``(enc_output, merged_hidden)`` where:
            - ``enc_output`` has shape ``(B, T, 2 * hidden_dim)``
            - ``merged_hidden`` has shape ``(B, hidden_dim)``
        """
        mask = input_ids.ne(self.pad_id)
        # (B, T) -> (B, T, C)
        embedded = self.embedding(input_ids)
        enhanced = self._apply_temporal_modules(embedded, mask)

        if torch.onnx.is_in_onnx_export():
            enc_output, hidden = self.gru(enhanced)
        else:
            lengths = mask.sum(dim=1).clamp(min=1).to(torch.long).cpu()
            packed = pack_padded_sequence(enhanced, lengths, batch_first=True, enforce_sorted=False)
            packed_output, hidden = self.gru(packed)
            enc_output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=input_ids.size(1),
            )

        batch_size = input_ids.size(0)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        top_layer_hidden = hidden[-1].transpose(0, 1).reshape(batch_size, self.hidden_dim * 2)
        merged_hidden = torch.tanh(self.hidden_projection(top_layer_hidden))
        return enc_output, merged_hidden

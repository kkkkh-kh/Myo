from typing import List, Optional, Tuple

import torch
from torch import nn

from model.attention import BahdanauAttention


class ChineseDecoder(nn.Module):
    """Attention-based GRU decoder for Chinese generation."""

    def __init__(
        self,
        zh_vocab_size: int = 8000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.zh_vocab_size = zh_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_id = pad_id
        self.enc_dim = hidden_dim * 2

        self.embedding = nn.Embedding(zh_vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(enc_dim=self.enc_dim, hidden_dim=hidden_dim)
        self.gru = nn.GRU(
            input_size=embed_dim + self.enc_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_bottleneck = nn.Linear(hidden_dim + self.enc_dim + embed_dim, embed_dim)
        self.vocab_projection = nn.Linear(embed_dim, zh_vocab_size, bias=False)
        self.vocab_projection.weight = self.embedding.weight
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.output_bottleneck.weight)
        nn.init.zeros_(self.output_bottleneck.bias)

    def init_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        return encoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)

    def forward_step(
        self,
        input_tokens: torch.Tensor,
        hidden: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_tokens)).unsqueeze(1)
        context, attn_weights = self.attention(enc_output, hidden[-1], mask=src_mask)
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, next_hidden = self.gru(gru_input, hidden)
        output = output.squeeze(1)
        features = torch.cat([output, context, embedded.squeeze(1)], dim=-1)
        bottleneck = torch.tanh(self.output_bottleneck(features))
        logits = self.vocab_projection(bottleneck)
        return logits, next_hidden, attn_weights

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
        teacher_forcing_ratio: float = 1.0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        hidden = self.init_hidden(encoder_hidden)
        batch_size = enc_output.size(0)

        if target_tokens is not None:
            steps = max(1, target_tokens.size(1) - 1)
            input_tokens = target_tokens[:, 0]
        else:
            steps = int(max_len or 1)
            input_tokens = torch.full(
                (batch_size,),
                bos_id,
                dtype=torch.long,
                device=enc_output.device,
            )

        logits_history = []
        predictions = []
        attn_history: List[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=enc_output.device)

        for step in range(steps):
            logits, hidden, attn_weights = self.forward_step(input_tokens, hidden, enc_output, src_mask)
            next_tokens = logits.argmax(dim=-1)
            logits_history.append(logits)
            predictions.append(next_tokens)
            attn_history.append(attn_weights)

            if target_tokens is not None and step + 1 < target_tokens.size(1):
                use_teacher_forcing = torch.rand(1, device=enc_output.device).item() < teacher_forcing_ratio
                input_tokens = target_tokens[:, step + 1] if use_teacher_forcing else next_tokens
            else:
                input_tokens = next_tokens
                finished = finished | next_tokens.eq(eos_id)
                if finished.all():
                    break

        logits_tensor = torch.stack(logits_history, dim=1)
        prediction_tensor = torch.stack(predictions, dim=1)
        return logits_tensor, prediction_tensor, attn_history

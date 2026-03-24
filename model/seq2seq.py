from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder


class Seq2Seq(nn.Module):
    """Full sequence-to-sequence model for gloss translation."""

    def __init__(
        self,
        encoder: GlossEncoder,
        decoder: ChineseDecoder,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def forward(
        self,
        gloss_ids: torch.Tensor,
        zh_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        enc_output, encoder_hidden = self.encoder(gloss_ids)
        src_mask = gloss_ids.ne(self.pad_id)
        logits, _, _ = self.decoder(
            encoder_hidden=encoder_hidden,
            enc_output=enc_output,
            src_mask=src_mask,
            target_tokens=zh_ids,
            teacher_forcing_ratio=teacher_forcing_ratio,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )
        return logits

    def _pad_predictions(self, predictions: torch.Tensor, max_len: int) -> torch.Tensor:
        if predictions.size(1) >= max_len:
            return predictions[:, :max_len]
        pad = torch.full(
            (predictions.size(0), max_len - predictions.size(1)),
            self.pad_id,
            dtype=predictions.dtype,
            device=predictions.device,
        )
        return torch.cat([predictions, pad], dim=1)

    def _greedy_translate(self, gloss_ids: torch.Tensor, max_len: int) -> torch.Tensor:
        enc_output, encoder_hidden = self.encoder(gloss_ids)
        src_mask = gloss_ids.ne(self.pad_id)
        _, predictions, _ = self.decoder(
            encoder_hidden=encoder_hidden,
            enc_output=enc_output,
            src_mask=src_mask,
            target_tokens=None,
            max_len=max_len,
            teacher_forcing_ratio=0.0,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )
        return self._pad_predictions(predictions, max_len)

    def _beam_translate(self, gloss_ids: torch.Tensor, max_len: int, beam_size: int) -> torch.Tensor:
        if gloss_ids.size(0) != 1:
            return self._greedy_translate(gloss_ids, max_len)

        enc_output, encoder_hidden = self.encoder(gloss_ids)
        src_mask = gloss_ids.ne(self.pad_id)
        hidden = self.decoder.init_hidden(encoder_hidden)
        beams = [([self.bos_id], hidden, 0.0)]

        for _ in range(max_len):
            candidates = []
            for sequence, beam_hidden, score in beams:
                last_token = sequence[-1]
                if last_token == self.eos_id:
                    candidates.append((sequence, beam_hidden, score))
                    continue
                input_token = torch.tensor([last_token], device=gloss_ids.device)
                logits, next_hidden, _ = self.decoder.forward_step(input_token, beam_hidden, enc_output, src_mask)
                log_probs = F.log_softmax(logits, dim=-1)
                top_scores, top_indices = log_probs.topk(beam_size, dim=-1)
                for token_score, token_index in zip(top_scores[0], top_indices[0]):
                    candidates.append(
                        (sequence + [int(token_index.item())], next_hidden, score + float(token_score.item()))
                    )
            beams = sorted(
                candidates,
                key=lambda item: item[2] / max(1, len(item[0]) - 1),
                reverse=True,
            )[:beam_size]
            if all(sequence[-1] == self.eos_id for sequence, _, _ in beams):
                break

        best_sequence = beams[0][0][1:]
        if not best_sequence:
            best_sequence = [self.eos_id]
        prediction = torch.tensor(best_sequence, dtype=torch.long, device=gloss_ids.device).unsqueeze(0)
        return self._pad_predictions(prediction, max_len)

    def translate(self, gloss_ids: torch.Tensor, max_len: int, beam_size: int = 1) -> torch.Tensor:
        if beam_size <= 1:
            return self._greedy_translate(gloss_ids, max_len)
        return self._beam_translate(gloss_ids, max_len, beam_size)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

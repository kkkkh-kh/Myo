from typing import List, Tuple

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
        return_attention: bool = False,
        current_epoch: int = 0,
        total_epochs: int = 80,
    ):
        enc_output, encoder_hidden = self.encoder(gloss_ids)
        src_mask = gloss_ids.ne(self.pad_id)
        logits, _, attention_history = self.decoder(
            encoder_hidden=encoder_hidden,
            enc_output=enc_output,
            src_mask=src_mask,
            target_tokens=zh_ids,
            teacher_forcing_ratio=teacher_forcing_ratio,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
        )
        if not return_attention:
            return logits
        attention_weights = self._stack_attention(attention_history, logits, gloss_ids.size(1))
        return logits, attention_weights

    def _stack_attention(
        self,
        attention_history: List[torch.Tensor],
        logits: torch.Tensor,
        src_len: int,
    ) -> torch.Tensor:
        if attention_history:
            return torch.stack(attention_history, dim=1)
        return logits.new_zeros((logits.size(0), 0, src_len))

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

    def _apply_generation_constraints(self, logits: torch.Tensor, step: int, min_len: int) -> torch.Tensor:
        constrained = logits.clone()
        constrained[:, self.pad_id] = float("-inf")
        constrained[:, self.bos_id] = float("-inf")
        if step < max(0, min_len):
            constrained[:, self.eos_id] = float("-inf")
        return constrained

    def _decoder_forward_step(
        self,
        input_tokens: torch.Tensor,
        hidden: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            return self.decoder.forward_step(
                input_tokens,
                hidden,
                enc_output,
                src_mask,
                current_step=step,
                total_steps=max(1, total_steps),
            )
        except TypeError:
            return self.decoder.forward_step(input_tokens, hidden, enc_output, src_mask)

    def _greedy_translate(self, gloss_ids: torch.Tensor, max_len: int, min_len: int = 1) -> torch.Tensor:
        enc_output, encoder_hidden = self.encoder(gloss_ids)
        src_mask = gloss_ids.ne(self.pad_id)
        hidden = self.decoder.init_hidden(encoder_hidden)
        batch_size = gloss_ids.size(0)
        input_tokens = torch.full((batch_size,), self.bos_id, dtype=torch.long, device=gloss_ids.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=gloss_ids.device)
        predictions = []

        for step in range(max_len):
            logits, hidden, _ = self._decoder_forward_step(
                input_tokens=input_tokens,
                hidden=hidden,
                enc_output=enc_output,
                src_mask=src_mask,
                step=step,
                total_steps=max_len,
            )
            logits = self._apply_generation_constraints(logits, step, min_len=min_len)
            next_tokens = logits.argmax(dim=-1)
            next_tokens = torch.where(finished, torch.full_like(next_tokens, self.pad_id), next_tokens)
            predictions.append(next_tokens)
            finished = finished | next_tokens.eq(self.eos_id)
            input_tokens = torch.where(finished, torch.full_like(next_tokens, self.eos_id), next_tokens)
            if finished.all():
                break

        prediction_tensor = torch.stack(predictions, dim=1)
        return self._pad_predictions(prediction_tensor, max_len)

    def _beam_translate(self, gloss_ids: torch.Tensor, max_len: int, beam_size: int, min_len: int = 1) -> torch.Tensor:
        if gloss_ids.size(0) != 1:
            raise ValueError("_beam_translate expects batch size 1")

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
                logits, next_hidden, _ = self._decoder_forward_step(
                    input_tokens=input_token,
                    hidden=beam_hidden,
                    enc_output=enc_output,
                    src_mask=src_mask,
                    step=max(0, len(sequence) - 1),
                    total_steps=max_len,
                )
                logits = self._apply_generation_constraints(logits, step=max(0, len(sequence) - 1), min_len=min_len)
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

    def translate(self, gloss_ids: torch.Tensor, max_len: int, beam_size: int = 1, min_len: int = 1) -> torch.Tensor:
        if beam_size <= 1:
            return self._greedy_translate(gloss_ids, max_len, min_len=min_len)
        if gloss_ids.size(0) == 1:
            return self._beam_translate(gloss_ids, max_len, beam_size, min_len=min_len)

        # Keep behavior consistent for batch decoding by running beam search per sample.
        predictions = [
            self._beam_translate(gloss_ids[index : index + 1], max_len, beam_size, min_len=min_len)
            for index in range(gloss_ids.size(0))
        ]
        return torch.cat(predictions, dim=0)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

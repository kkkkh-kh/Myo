import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder
from model.seq2seq import Seq2Seq


class StubEncoder(nn.Module):
    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        enc_output = torch.zeros(batch_size, seq_len, 4)
        merged_hidden = torch.zeros(batch_size, 2)
        return enc_output, merged_hidden


class StubDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def init_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, encoder_hidden.size(0), encoder_hidden.size(1))

    def forward_step(self, input_tokens, hidden, enc_output, src_mask):
        del input_tokens, src_mask
        batch_size = enc_output.size(0)
        logits = torch.full((batch_size, 6), -10.0)
        if self.calls == 0:
            logits[:, 2] = 5.0
            logits[:, 1] = 4.0
            logits[:, 0] = 3.0
            logits[:, 4] = 2.0
        else:
            logits[:, 2] = 5.0
        self.calls += 1
        attn = torch.ones(batch_size, enc_output.size(1)) / max(1, enc_output.size(1))
        return logits, hidden, attn


class ModelTestCase(unittest.TestCase):
    def test_forward_shapes_and_parameter_budget(self):
        encoder = GlossEncoder(gloss_vocab_size=4000, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3)
        decoder = ChineseDecoder(zh_vocab_size=8000, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3)
        model = Seq2Seq(encoder=encoder, decoder=decoder)

        gloss_ids = torch.randint(0, 128, (2, 10), dtype=torch.long)
        zh_ids = torch.randint(0, 128, (2, 12), dtype=torch.long)
        zh_ids[:, 0] = 1
        logits = model(gloss_ids, zh_ids, teacher_forcing_ratio=0.5)
        predictions = model.translate(gloss_ids, max_len=8)

        self.assertEqual(logits.shape, (2, 11, 8000))
        self.assertEqual(predictions.shape, (2, 8))
        self.assertLessEqual(model.count_parameters(), 5_000_000)

    def test_translate_suppresses_special_tokens_before_first_content_token(self):
        model = Seq2Seq(encoder=StubEncoder(), decoder=StubDecoder(), pad_id=0, bos_id=1, eos_id=2)
        gloss_ids = torch.tensor([[3, 4, 2]], dtype=torch.long)

        predictions = model.translate(gloss_ids, max_len=4)

        self.assertEqual(predictions.shape, (1, 4))
        self.assertEqual(predictions[0, 0].item(), 4)
        self.assertEqual(predictions[0, 1].item(), 2)
        self.assertEqual(predictions[0, 2].item(), 0)
        self.assertEqual(predictions[0, 3].item(), 0)


if __name__ == "__main__":
    unittest.main()


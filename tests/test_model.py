import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder
from model.seq2seq import Seq2Seq


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


if __name__ == "__main__":
    unittest.main()

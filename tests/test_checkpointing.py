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
from train.checkpointing import load_checkpoint_into_model, model_is_qat_prepared, prepare_model_for_qat


class CheckpointingTestCase(unittest.TestCase):
    def test_load_checkpoint_into_model_prepares_qat_when_needed(self):
        qat_model = Seq2Seq(
            encoder=GlossEncoder(gloss_vocab_size=32, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0),
            decoder=ChineseDecoder(zh_vocab_size=32, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0),
        )
        prepare_model_for_qat(qat_model)
        checkpoint = {"model_state_dict": qat_model.state_dict()}

        fresh_model = Seq2Seq(
            encoder=GlossEncoder(gloss_vocab_size=32, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0),
            decoder=ChineseDecoder(zh_vocab_size=32, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0),
        )
        self.assertFalse(model_is_qat_prepared(fresh_model))

        load_checkpoint_into_model(fresh_model, checkpoint)

        self.assertTrue(model_is_qat_prepared(fresh_model))
        for name, parameter in fresh_model.state_dict().items():
            self.assertEqual(parameter.shape, checkpoint["model_state_dict"][name].shape)


if __name__ == "__main__":
    unittest.main()


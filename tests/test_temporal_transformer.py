import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from modules.temporal_transformer import TemporalTransformerEncoder


class TemporalTransformerEncoderTestCase(unittest.TestCase):
    def test_preserves_shape_and_respects_padding_mask(self):
        module = TemporalTransformerEncoder(d_model=32, num_layers=2, num_heads=4, dropout=0.0)
        x = torch.randn(2, 6, 32)
        mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
            ]
        )

        output = module(x, padding_mask=mask)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.allclose(output[0, 3:], torch.zeros_like(output[0, 3:]), atol=1e-6))
        self.assertTrue(torch.allclose(output[1, 4:], torch.zeros_like(output[1, 4:]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()

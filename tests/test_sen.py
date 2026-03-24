import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from modules.sen import TemporalSEN


class TemporalSENTestCase(unittest.TestCase):
    def test_preserves_shape_and_dtype(self):
        module = TemporalSEN(channels=32, reduction=8)
        x = torch.randn(3, 7, 32, dtype=torch.float32)
        output = module(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from train.evaluate import compute_rouge_l, compute_wer


class EvaluateMetricsTestCase(unittest.TestCase):
    def test_rouge_l_handles_tokenized_chinese_sequences(self):
        hypothesis = "我 们 的 时间"
        reference = "我 们 的 课程"

        score = compute_rouge_l([hypothesis], [reference])

        self.assertGreater(score, 50.0)

    def test_wer_uses_token_boundaries_for_spaced_sequences(self):
        score = compute_wer(["我 们 的"], ["我 们 的 时间"])
        self.assertAlmostEqual(score, 25.0)


if __name__ == "__main__":
    unittest.main()


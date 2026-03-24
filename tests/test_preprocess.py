import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from data.preprocess import tokenize_chinese
from modules.postprocess import PostProcessor


class ChineseTokenizationTestCase(unittest.TestCase):
    def test_char_tokenization_keeps_single_char_targets(self):
        tokens = tokenize_chinese("我今天去学校。", mode="char")
        self.assertEqual(tokens, ["我", "今", "天", "去", "学", "校", "."])

    def test_jieba_mode_is_still_available(self):
        tokens = tokenize_chinese("我今天去学校。", mode="jieba")
        self.assertGreaterEqual(len(tokens), 3)

    def test_postprocessor_handles_character_level_output(self):
        postprocess = PostProcessor()
        sentence = postprocess.process(["我", "今", "天", "去", "学", "校", "."])
        self.assertTrue(sentence)


if __name__ == "__main__":
    unittest.main()

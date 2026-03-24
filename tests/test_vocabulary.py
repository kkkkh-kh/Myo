import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from data.vocabulary import Vocabulary


class VocabularyTestCase(unittest.TestCase):
    def test_build_encode_decode_and_fallback(self):
        vocab = Vocabulary()
        vocab.build_from_corpus([["我", "昨天", "买", "苹果"], ["你", "今天", "学习"]], max_size=16)
        encoded = vocab.encode(["我", "苹果"], add_bos=True, add_eos=True)
        decoded = vocab.decode(encoded)
        self.assertIn("我", decoded)
        self.assertIn("苹果", decoded)
        fallback_ids = vocab.encode(["未知词"])
        self.assertGreaterEqual(len(fallback_ids), 1)

    def test_save_and_load(self):
        vocab = Vocabulary()
        vocab.build_from_corpus([["残疾人", "申请", "补偿"]], max_size=12)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vocab.json"
            vocab.save(path)
            loaded = Vocabulary.load(path)
            self.assertEqual(len(vocab), len(loaded))
            self.assertEqual(vocab.encode(["残疾人", "申请"]), loaded.encode(["残疾人", "申请"]))


if __name__ == "__main__":
    unittest.main()

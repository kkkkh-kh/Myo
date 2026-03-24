import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from modules.preorder import PreorderModule


class PreorderModuleTestCase(unittest.TestCase):
    def setUp(self):
        self.module = PreorderModule()

    def test_reorder_subject_object_verb(self):
        tokens = ["我", "苹果", "买"]
        reordered = self.module.reorder(tokens)
        self.assertEqual(reordered, ["我", "买", "苹果"])

    def test_handles_empty_and_single_token(self):
        self.assertEqual(self.module.reorder([]), [])
        self.assertEqual(self.module.reorder(["我"]), ["我"])


if __name__ == "__main__":
    unittest.main()

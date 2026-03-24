import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from data.gloss_noise_augment import GlossNoiseAugmentor
from data.vocabulary import Vocabulary


class GlossNoiseAugmentorTestCase(unittest.TestCase):
    def test_special_tokens_are_preserved(self):
        augmentor = GlossNoiseAugmentor(
            candidate_token_ids=list(range(4, 16)),
            p_del=1.0,
            p_sub=1.0,
            p_ins=1.0,
            p_rep=1.0,
            seed=7,
        )
        token_ids = [Vocabulary.BOS_ID, 4, 5, 6, Vocabulary.EOS_ID]

        augmented = augmentor(token_ids, epoch=5, total_epochs=10)

        self.assertEqual(augmented[0], Vocabulary.BOS_ID)
        self.assertEqual(augmented[-1], Vocabulary.EOS_ID)
        self.assertEqual(augmented.count(Vocabulary.BOS_ID), 1)
        self.assertEqual(augmented.count(Vocabulary.EOS_ID), 1)

    def test_warmup_halves_noise_scale(self):
        augmentor = GlossNoiseAugmentor(candidate_token_ids=list(range(4, 16)), warmup_ratio=0.2)

        self.assertEqual(augmentor.probability_scale(epoch=0, total_epochs=10), 0.5)
        self.assertEqual(augmentor.probability_scale(epoch=1, total_epochs=10), 0.5)
        self.assertEqual(augmentor.probability_scale(epoch=2, total_epochs=10), 1.0)


if __name__ == "__main__":
    unittest.main()

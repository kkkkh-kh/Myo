# Module description: gloss token noise augmentation for robust gloss-to-text translation training.

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence

from data.vocabulary import Vocabulary


class GlossNoiseAugmentor:
    """Apply deletion, substitution, insertion, and repetition noise to gloss ids.

    The augmentor operates on token-id sequences before padding so that training can
    simulate upstream gloss recognition errors while keeping special tokens intact.

    Args:
        candidate_token_ids: Non-special token ids that can be sampled for insertion
            and substitution.
        p_del: Probability of deleting a non-special token.
        p_sub: Probability of substituting a non-special token.
        p_ins: Probability of inserting a sampled token before a non-special token.
        p_rep: Probability of repeating a non-special token once.
        warmup_ratio: Fraction of total epochs during which the effective noise
            probabilities are halved.
        pad_id: Padding token id.
        bos_id: Beginning-of-sequence token id.
        eos_id: End-of-sequence token id.
        seed: Optional RNG seed for deterministic tests.
    """

    def __init__(
        self,
        candidate_token_ids: Sequence[int],
        p_del: float = 0.05,
        p_sub: float = 0.05,
        p_ins: float = 0.03,
        p_rep: float = 0.03,
        warmup_ratio: float = 0.2,
        pad_id: int = Vocabulary.PAD_ID,
        bos_id: int = Vocabulary.BOS_ID,
        eos_id: int = Vocabulary.EOS_ID,
        seed: Optional[int] = None,
    ) -> None:
        for name, value in {
            "p_del": p_del,
            "p_sub": p_sub,
            "p_ins": p_ins,
            "p_rep": p_rep,
            "warmup_ratio": warmup_ratio,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.p_del = p_del
        self.p_sub = p_sub
        self.p_ins = p_ins
        self.p_rep = p_rep
        self.warmup_ratio = warmup_ratio
        self.rng = random.Random(seed)
        self.protected_ids = {pad_id, bos_id, eos_id}
        self.candidate_token_ids = [token_id for token_id in candidate_token_ids if token_id not in self.protected_ids]
        self.current_epoch = 0
        self.total_epochs: Optional[int] = None

    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Update the current epoch so warmup scaling can be computed dynamically."""
        self.current_epoch = max(0, int(epoch))
        if total_epochs is not None:
            self.total_epochs = max(1, int(total_epochs))

    def probability_scale(self, epoch: Optional[int] = None, total_epochs: Optional[int] = None) -> float:
        """Return the current probability multiplier for warmup scheduling."""
        if self.warmup_ratio <= 0.0:
            return 1.0

        resolved_total_epochs = total_epochs if total_epochs is not None else self.total_epochs
        if resolved_total_epochs is None or resolved_total_epochs <= 0:
            return 1.0

        resolved_epoch = self.current_epoch if epoch is None else max(0, int(epoch))
        warmup_epochs = max(1, int(math.ceil(resolved_total_epochs * self.warmup_ratio)))
        return 0.5 if resolved_epoch < warmup_epochs else 1.0

    def _sample_token_id(self) -> Optional[int]:
        if not self.candidate_token_ids:
            return None
        return self.rng.choice(self.candidate_token_ids)

    def _is_protected(self, token_id: int) -> bool:
        return token_id in self.protected_ids

    def augment(
        self,
        token_ids: Sequence[int],
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> List[int]:
        """Return a noise-augmented copy of the input token-id sequence."""
        scale = self.probability_scale(epoch=epoch, total_epochs=total_epochs)
        p_del = self.p_del * scale
        p_sub = self.p_sub * scale
        p_ins = self.p_ins * scale
        p_rep = self.p_rep * scale

        augmented: List[int] = []
        for token_id in token_ids:
            if self._is_protected(token_id):
                augmented.append(token_id)
                continue

            inserted = self._sample_token_id()
            if inserted is not None and self.rng.random() < p_ins:
                augmented.append(inserted)

            if self.rng.random() < p_del:
                continue

            current_token = token_id
            replacement = self._sample_token_id()
            if replacement is not None and self.rng.random() < p_sub:
                current_token = replacement

            augmented.append(current_token)
            if self.rng.random() < p_rep:
                augmented.append(current_token)

        if not augmented:
            return list(token_ids)
        return augmented

    def __call__(
        self,
        token_ids: Sequence[int],
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> List[int]:
        """Alias for :meth:`augment` so the class can be used as a callable."""
        return self.augment(token_ids, epoch=epoch, total_epochs=total_epochs)


if __name__ == "__main__":
    augmentor = GlossNoiseAugmentor(candidate_token_ids=list(range(4, 16)), p_del=0.2, p_sub=0.2, p_ins=0.1, p_rep=0.1, seed=7)
    sample = [Vocabulary.BOS_ID, 4, 5, 6, Vocabulary.EOS_ID]
    print("input :", sample)
    print("output:", augmentor(sample, epoch=0, total_epochs=10))

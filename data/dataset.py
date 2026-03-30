from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from data.gloss_noise_augment import GlossNoiseAugmentor
from data.preprocess import read_parallel_pairs, tokenize_chinese, tokenize_gloss
from data.vocabulary import Vocabulary
from modules.preorder import PreorderModule


class GlossChineseDataset(Dataset):
    """Dataset for gloss-to-Chinese sequence modeling."""

    def __init__(
        self,
        tsv_path: str,
        gloss_vocab: Vocabulary,
        zh_vocab: Vocabulary,
        max_gloss_len: int = 32,
        max_zh_len: int = 48,
        preorder_module: Optional[PreorderModule] = None,
        transform: Optional[Callable[[List[str], List[str]], Tuple[List[str], List[str]]]] = None,
        data_path: Optional[str] = None,
        zh_tokenizer_mode: str = "char",
        augment: bool = False,
        augmentor: Optional[GlossNoiseAugmentor] = None,
    ) -> None:
        resolved_path = data_path or tsv_path
        if not resolved_path:
            raise ValueError("A dataset path must be provided.")
        self.data_path = Path(resolved_path)
        self.gloss_vocab = gloss_vocab
        self.zh_vocab = zh_vocab
        self.max_gloss_len = max_gloss_len
        self.max_zh_len = max_zh_len
        self.preorder_module = preorder_module or PreorderModule()
        self.transform = transform
        self.zh_tokenizer_mode = zh_tokenizer_mode
        self.augment = augment
        self.augmentor = augmentor
        self.current_epoch = 0
        self.total_epochs: Optional[int] = None
        self.samples = self._load_samples()

    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Update the dataset epoch so augmentation warmup can follow training."""
        self.current_epoch = max(0, int(epoch))
        if total_epochs is not None:
            self.total_epochs = max(1, int(total_epochs))
        if self.augmentor is not None:
            self.augmentor.set_epoch(self.current_epoch, total_epochs=self.total_epochs)

    def _pad_ids(
        self,
        ids: Sequence[int],
        max_len: int,
        pad_id: int,
        preserve_last_eos: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        truncated = list(ids[:max_len])
        if not truncated:
            truncated = [Vocabulary.EOS_ID]
        if preserve_last_eos and ids and ids[-1] == Vocabulary.EOS_ID and truncated[-1] != Vocabulary.EOS_ID:
            truncated[-1] = Vocabulary.EOS_ID
        length = len(truncated)
        if length < max_len:
            truncated.extend([pad_id] * (max_len - length))
        return torch.tensor(truncated, dtype=torch.long), length

    def _prepare_target_ids(self, tokens: List[str]) -> Tuple[torch.Tensor, int]:
        token_ids = self.zh_vocab.encode(tokens, add_bos=True, add_eos=True)
        return self._pad_ids(token_ids, self.max_zh_len, Vocabulary.PAD_ID, preserve_last_eos=True)

    def _load_samples(self) -> List[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        samples: List[Tuple[List[int], torch.Tensor, torch.Tensor]] = []
        for gloss_text, chinese_text in read_parallel_pairs(self.data_path.as_posix()):
            gloss_tokens = tokenize_gloss(gloss_text)
            zh_tokens = tokenize_chinese(chinese_text, mode=self.zh_tokenizer_mode)
            gloss_tokens = self.preorder_module.reorder(gloss_tokens)

            if self.transform is not None:
                gloss_tokens, zh_tokens = self.transform(gloss_tokens, zh_tokens)

            gloss_ids = self.gloss_vocab.encode(gloss_tokens, add_eos=True)
            zh_ids, zh_length = self._prepare_target_ids(zh_tokens)
            samples.append((gloss_ids, zh_ids, torch.tensor(zh_length, dtype=torch.long)))
        return samples

    def _prepare_gloss_item(self, gloss_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = list(gloss_ids)
        if self.augment and self.augmentor is not None:
            token_ids = self.augmentor(token_ids, epoch=self.current_epoch, total_epochs=self.total_epochs)
        gloss_tensor, gloss_length = self._pad_ids(
            token_ids,
            self.max_gloss_len,
            Vocabulary.PAD_ID,
            preserve_last_eos=True,
        )
        return gloss_tensor, torch.tensor(gloss_length, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gloss_ids, zh_ids, zh_length = self.samples[index]
        gloss_tensor, gloss_length = self._prepare_gloss_item(gloss_ids)
        return gloss_tensor, gloss_length, zh_ids, zh_length

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gloss_ids = torch.stack([item[0] for item in batch], dim=0)
        gloss_lengths = torch.stack([item[1] for item in batch], dim=0)
        zh_ids = torch.stack([item[2] for item in batch], dim=0)
        zh_lengths = torch.stack([item[3] for item in batch], dim=0)
        return gloss_ids, gloss_lengths, zh_ids, zh_lengths

import csv
import json
import math
import shutil
import sys
import unittest
import uuid
from pathlib import Path

from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from data.dataset import GlossChineseDataset
from data.preprocess import extract_corpora, read_parallel_pairs
from data.vocabulary import Vocabulary
from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder
from model.seq2seq import Seq2Seq
from train.trainer import Trainer


class TrainerLoggingTestCase(unittest.TestCase):
    def _build_minimal_components(self, base_dir: Path):
        dataset_path = base_dir / "train.csv"
        dataset_path.write_text(
            "\n".join(
                [
                    "Number,Translator,Chinese Sentences,Gloss,Note",
                    "train-00001,A,我今天去学校,我/今天/去/学校,",
                    "train-00002,A,他昨天买苹果,他/昨天/买/苹果,",
                    "train-00003,A,老师说明作业,老师/说明/作业,",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        pairs = read_parallel_pairs(dataset_path.as_posix())
        gloss_corpus, zh_corpus = extract_corpora(pairs)
        gloss_vocab = Vocabulary()
        gloss_vocab.build_from_corpus(gloss_corpus, max_size=64)
        zh_vocab = Vocabulary()
        zh_vocab.build_from_corpus(zh_corpus, max_size=64)

        dataset = GlossChineseDataset(
            tsv_path=dataset_path.as_posix(),
            gloss_vocab=gloss_vocab,
            zh_vocab=zh_vocab,
            max_gloss_len=16,
            max_zh_len=16,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=GlossChineseDataset.collate_fn)

        encoder = GlossEncoder(gloss_vocab_size=len(gloss_vocab), embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
        decoder = ChineseDecoder(zh_vocab_size=len(zh_vocab), embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
        model = Seq2Seq(encoder=encoder, decoder=decoder)
        return loader, model, len(zh_vocab)

    def test_training_writes_extended_validation_logs(self):
        base_dir = WORKSPACE_ROOT / f"tmp_trainer_test_{uuid.uuid4().hex}"
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            loader, model, zh_vocab_size = self._build_minimal_components(base_dir)
            config = {
                "run_id": "trainer_test_run",
                "save_dir": base_dir.as_posix(),
                "model": {"zh_vocab_size": zh_vocab_size},
                "train": {
                    "epochs": 1,
                    "label_smoothing": 0.05,
                    "teacher_forcing_ratio_start": 1.0,
                    "teacher_forcing_ratio_end": 0.5,
                    "teacher_forcing_decay_epochs": 12,
                    "clip_grad_norm": 1.0,
                    "early_stopping_patience": 1,
                    "validation_sample_size": 2,
                    "validation_beam_size": 1,
                    "qat_enabled": False,
                },
            }

            trainer = Trainer(model=model, optimizer=AdamW(model.parameters(), lr=1e-3), scheduler=None, config=config)
            result = trainer.train(loader, loader)

            self.assertTrue(math.isfinite(result["best_val_loss"]))

            log_path = base_dir / "training_log.csv"
            with log_path.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)

            self.assertEqual(
                reader.fieldnames,
                [
                    "run_id",
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_bleu4",
                    "val_rouge_l",
                    "val_wer",
                    "teacher_forcing_ratio",
                ],
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "trainer_test_run")

            samples_path = base_dir / "validation_samples.jsonl"
            with samples_path.open("r", encoding="utf-8") as file:
                record = json.loads(file.readline())

            self.assertEqual(record["run_id"], "trainer_test_run")
            self.assertEqual(record["epoch"], 1)
            self.assertIn("val_rouge_l", record)
            self.assertIn("val_wer", record)
            self.assertLessEqual(len(record["samples"]), 2)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_qat_is_opt_in_when_flag_is_missing(self):
        base_dir = WORKSPACE_ROOT / f"tmp_trainer_qat_test_{uuid.uuid4().hex}"
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            loader, model, zh_vocab_size = self._build_minimal_components(base_dir)
            config = {
                "run_id": "trainer_qat_default_off",
                "save_dir": base_dir.as_posix(),
                "model": {"zh_vocab_size": zh_vocab_size},
                "train": {
                    "epochs": 2,
                    "label_smoothing": 0.05,
                    "teacher_forcing_ratio_start": 1.0,
                    "teacher_forcing_ratio_end": 0.5,
                    "teacher_forcing_decay_epochs": 12,
                    "clip_grad_norm": 1.0,
                    "early_stopping_patience": 1,
                    "validation_sample_size": 1,
                    "validation_beam_size": 1,
                },
            }

            trainer = Trainer(model=model, optimizer=AdamW(model.parameters(), lr=1e-3), scheduler=None, config=config)
            trainer.current_epoch = 0
            trainer._prepare_qat_if_needed()

            self.assertFalse(trainer.qat_prepared)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


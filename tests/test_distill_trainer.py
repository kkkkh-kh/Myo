import shutil
import sys
import unittest
import uuid
from pathlib import Path

import torch
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
from train.distill_trainer import DistillTrainer


class DistillTrainerTestCase(unittest.TestCase):
    def test_minimal_distillation_flow_runs(self):
        temp_dir = WORKSPACE_ROOT / f"tmp_distill_test_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            dataset_path = temp_dir / "train.csv"
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

            config = {
                "device": "cpu",
                "project_root": PROJECT_ROOT.as_posix(),
                "save_dir": temp_dir.as_posix(),
                "model": {
                    "gloss_vocab_size": len(gloss_vocab),
                    "zh_vocab_size": len(zh_vocab),
                    "embed_dim": 16,
                    "hidden_dim": 32,
                    "num_layers": 1,
                    "dropout": 0.0,
                },
                "encoder": {
                    "use_sen": False,
                    "sen_reduction": 16,
                    "use_transformer": False,
                    "transformer_layers": 2,
                    "transformer_heads": 4,
                    "transformer_dropout": 0.1,
                },
                "train": {
                    "epochs": 1,
                    "label_smoothing": 0.05,
                    "clip_grad_norm": 1.0,
                },
                "distillation": {
                    "teacher_path": (temp_dir / "best_model.pt").as_posix(),
                    "student_init": "hot_start",
                    "alpha": 0.5,
                    "temperature": 2.0,
                    "epochs": 1,
                    "lr": 1.0e-3,
                    "save_path": (temp_dir / "distilled_model.pt").as_posix(),
                },
            }

            teacher_encoder = GlossEncoder(
                gloss_vocab_size=len(gloss_vocab),
                embed_dim=16,
                hidden_dim=32,
                num_layers=1,
                dropout=0.0,
            )
            teacher_decoder = ChineseDecoder(
                zh_vocab_size=len(zh_vocab),
                embed_dim=16,
                hidden_dim=32,
                num_layers=1,
                dropout=0.0,
            )
            teacher_model = Seq2Seq(encoder=teacher_encoder, decoder=teacher_decoder)
            teacher_path = temp_dir / "best_model.pt"
            torch.save({"model_state_dict": teacher_model.state_dict(), "config": config}, teacher_path)

            student_encoder = GlossEncoder(
                gloss_vocab_size=len(gloss_vocab),
                embed_dim=16,
                hidden_dim=32,
                num_layers=1,
                dropout=0.0,
            )
            student_decoder = ChineseDecoder(
                zh_vocab_size=len(zh_vocab),
                embed_dim=16,
                hidden_dim=32,
                num_layers=1,
                dropout=0.0,
            )
            student_model = Seq2Seq(encoder=student_encoder, decoder=student_decoder)

            trainer = DistillTrainer(student_model=student_model, config=config)
            result = trainer.distill(loader, loader)

            self.assertTrue((temp_dir / "distilled_model.pt").exists())
            self.assertIn("best_bleu4", result)
            self.assertIn("best_rouge_l", result)
            self.assertIn("best_wer", result)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

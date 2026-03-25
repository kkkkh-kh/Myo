import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from train.plm_utils import (
    ApproxSemanticAwareLabelSmoother,
    format_gloss_prompt,
    load_training_examples,
    read_parallel_examples,
    source_special_token,
)


class PlmUtilsTestCase(unittest.TestCase):
    def test_format_gloss_prompt_can_include_source_tag(self):
        prompt = format_gloss_prompt(
            "I / YESTERDAY / BUY / APPLE",
            task_prefix="translate gloss to chinese:",
            source="silver gloss",
            include_source_tag=True,
        )
        self.assertIn("translate gloss to chinese:", prompt)
        self.assertIn(source_special_token("silver gloss"), prompt)
        self.assertIn("I YESTERDAY BUY APPLE", prompt)

    def test_jsonl_examples_and_synthetic_ratio_are_loaded(self):
        base_dir = WORKSPACE_ROOT / f"tmp_plm_utils_{uuid.uuid4().hex}"
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            train_path = base_dir / "train.csv"
            train_path.write_text(
                "\n".join(
                    [
                        "Number,Translator,Chinese Sentences,Gloss,Note",
                        "train-1,A,alpha sentence,I/TODAY/GO/SCHOOL,",
                        "train-2,A,beta sentence,HE/YESTERDAY/BUY/APPLE,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            synthetic_path = base_dir / "paraphrase.jsonl"
            records = [
                {"gloss": "I TODAY GO SCHOOL", "chinese": "alpha paraphrase one", "source": "paraphrase"},
                {"gloss": "HE YESTERDAY BUY APPLE", "chinese": "beta paraphrase one", "source": "paraphrase"},
                {"gloss": "TEACHER EXPLAIN HOMEWORK", "chinese": "gamma paraphrase one", "source": "paraphrase"},
            ]
            with synthetic_path.open("w", encoding="utf-8") as file:
                for record in records:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

            examples = load_training_examples(
                train_path.as_posix(),
                synthetic_paths=[synthetic_path.as_posix()],
                max_synthetic_ratio=0.5,
                seed=7,
            )

            self.assertEqual(len(examples), 3)
            self.assertEqual(sum(1 for example in examples if example.source == "real"), 2)
            self.assertEqual(sum(1 for example in examples if example.source == "paraphrase"), 1)
            self.assertEqual(len(read_parallel_examples(synthetic_path.as_posix())), 3)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)

    def test_semantic_smoother_uses_neighbor_mass(self):
        smoother = ApproxSemanticAwareLabelSmoother({1: [(2, 1.0)]}, smoothing=0.2, ignore_index=-100)
        logits = torch.tensor([[[0.0, 2.0, 1.5]]], dtype=torch.float32)
        labels = torch.tensor([[1]], dtype=torch.long)

        semantic_loss = float(smoother(logits, labels).item())
        nll_loss = float(torch.nn.functional.cross_entropy(logits.view(-1, 3), labels.view(-1)).item())

        self.assertNotEqual(round(semantic_loss, 6), round(nll_loss, 6))
        self.assertGreater(semantic_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
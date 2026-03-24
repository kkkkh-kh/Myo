import shutil
import sys
import unittest
import uuid
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from data.vocabulary import Vocabulary
from deploy.export_onnx import export_to_onnx
from deploy.memory_check import MemoryProfiler
from deploy.quantize import quantize_models
from inference.pipeline import TranslationPipeline
from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder
from model.seq2seq import Seq2Seq


TEST_GLOSS_INPUTS = [
    "我 昨天 买 苹果",
    "你 今天 去 学校",
    "他 明天 提交 材料",
    "残疾人 申请 政府 补偿",
    "学生 现在 学习 中文",
]


def _build_vocab(texts, max_size):
    vocab = Vocabulary()
    vocab.build_from_corpus(texts, max_size=max_size)
    return vocab


def _bootstrap_assets(base_dir: Path):
    torch.manual_seed(7)
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_dir / "default.yaml"

    gloss_texts = [item.split() for item in TEST_GLOSS_INPUTS]
    zh_texts = [
        ["我", "昨天", "买", "苹果"],
        ["你", "今天", "去", "学校"],
        ["他", "明天", "提交", "材料"],
        ["残疾人", "申请", "政府", "补偿"],
        ["学生", "现在", "学习", "中文"],
    ]
    gloss_vocab = _build_vocab(gloss_texts, max_size=128)
    zh_vocab = _build_vocab(zh_texts, max_size=256)
    gloss_vocab.save(model_dir / "gloss_vocab.json")
    zh_vocab.save(model_dir / "zh_vocab.json")

    config = {
        "model": {
            "gloss_vocab_size": max(128, len(gloss_vocab) + 8),
            "zh_vocab_size": max(256, len(zh_vocab) + 8),
            "embed_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.1,
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
            "batch_size": 4,
            "learning_rate": 3e-4,
            "epochs": 1,
            "teacher_forcing_ratio_start": 1.0,
            "teacher_forcing_ratio_end": 0.5,
            "teacher_forcing_decay_epochs": 1,
            "clip_grad_norm": 1.0,
            "label_smoothing": 0.1,
            "early_stopping_patience": 1,
            "qat_enabled": False,
        },
        "data": {"max_gloss_len": 16, "max_zh_len": 16, "train_split": 0.9, "val_split": 0.05, "test_split": 0.05},
        "deploy": {"quantization": "int8", "max_seq_len": 16, "memory_limit_mb": 60, "beam_size": 1, "use_distilled": True},
    }
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    encoder = GlossEncoder(
        gloss_vocab_size=config["model"]["gloss_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    decoder = ChineseDecoder(
        zh_vocab_size=config["model"]["zh_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    export_to_onnx(model, model_dir.as_posix())
    quantize_models(model_dir.as_posix(), model_dir.as_posix())
    return model_dir, config_path


def test_inference_memory_under_60mb():
    base_dir = WORKSPACE_ROOT / f"tmp_memory_test_{uuid.uuid4().hex}"
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        model_dir, config_path = _bootstrap_assets(base_dir)
        pipeline = TranslationPipeline(model_dir=model_dir.as_posix(), config_path=config_path.as_posix())
        profiler = MemoryProfiler()
        result = profiler.measure_inference_memory(pipeline, TEST_GLOSS_INPUTS)
        assert result["peak_mb"] <= 60.0, f"Memory exceeded: {result['peak_mb']:.1f}MB > 60MB"
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


class MemoryConstraintTestCase(unittest.TestCase):
    def test_inference_memory_under_60mb(self):
        test_inference_memory_under_60mb()


if __name__ == "__main__":
    unittest.main()


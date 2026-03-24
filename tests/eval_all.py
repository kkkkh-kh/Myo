# Module description: unified evaluation script for PyTorch and ONNX gloss-to-Chinese models.

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from data.dataset import GlossChineseDataset
from data.preprocess import read_parallel_pairs
from data.vocabulary import Vocabulary
from inference.pipeline import TranslationPipeline
from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder
from model.seq2seq import Seq2Seq
from train.evaluate import compute_bleu4, compute_rouge_l, compute_wer

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


def resolve_dataset_path(data_dir: Path, *candidates: str) -> Path:
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No dataset file found in {data_dir} for candidates: {candidates}")


def load_config(checkpoint_dir: Path, config_path: Optional[Path]) -> Dict:
    candidate = config_path or checkpoint_dir / "runtime_config.yaml"
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    fallback = PROJECT_ROOT / "configs" / "default.yaml"
    with fallback.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_model(config: Dict, checkpoint_dir: Path) -> Seq2Seq:
    gloss_vocab = Vocabulary.load(checkpoint_dir / "gloss_vocab.json")
    zh_vocab = Vocabulary.load(checkpoint_dir / "zh_vocab.json")
    config = dict(config)
    config.setdefault("model", {})
    config.setdefault("encoder", {})
    config["model"]["gloss_vocab_size"] = len(gloss_vocab)
    config["model"]["zh_vocab_size"] = len(zh_vocab)
    encoder_config = config.get("encoder", {})

    encoder = GlossEncoder(
        gloss_vocab_size=config["model"]["gloss_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        pad_id=Vocabulary.PAD_ID,
        use_sen=bool(encoder_config.get("use_sen", False)),
        sen_reduction=int(encoder_config.get("sen_reduction", 16)),
        use_transformer=bool(encoder_config.get("use_transformer", False)),
        transformer_layers=int(encoder_config.get("transformer_layers", 2)),
        transformer_heads=int(encoder_config.get("transformer_heads", 4)),
        transformer_dropout=float(encoder_config.get("transformer_dropout", 0.1)),
    )
    decoder = ChineseDecoder(
        zh_vocab_size=config["model"]["zh_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    return Seq2Seq(encoder=encoder, decoder=decoder)


def build_test_loader(checkpoint_dir: Path, data_path: Path, config: Dict) -> DataLoader:
    gloss_vocab = Vocabulary.load(checkpoint_dir / "gloss_vocab.json")
    zh_vocab = Vocabulary.load(checkpoint_dir / "zh_vocab.json")
    dataset = GlossChineseDataset(
        tsv_path=data_path.as_posix(),
        gloss_vocab=gloss_vocab,
        zh_vocab=zh_vocab,
        max_gloss_len=int(config.get("data", {}).get("max_gloss_len", 32)),
        max_zh_len=int(config.get("data", {}).get("max_zh_len", 48)),
        zh_tokenizer_mode=config.get("data", {}).get("zh_tokenizer", "char"),
        augment=False,
        augmentor=None,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=GlossChineseDataset.collate_fn)


def _process() -> Optional["psutil.Process"]:
    if psutil is None:
        return None
    return psutil.Process()


def evaluate_pytorch_model(checkpoint_path: Path, checkpoint_dir: Path, config: Dict, data_path: Path) -> Dict[str, object]:
    model = build_model(config, checkpoint_dir)
    state = torch.load(checkpoint_path.as_posix(), map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    loader = build_test_loader(checkpoint_dir, data_path, config)
    dataset = loader.dataset
    zh_vocab = dataset.zh_vocab

    hypotheses: List[str] = []
    references: List[str] = []
    process = _process()
    peak_memory_mb = process.memory_info().rss / (1024 * 1024) if process is not None else None
    start = time.perf_counter()
    with torch.no_grad():
        for gloss_ids, _, zh_ids, _ in loader:
            predictions = model.translate(gloss_ids, max_len=zh_ids.size(1) - 1)
            for predicted_ids, reference_ids in zip(predictions, zh_ids):
                hypotheses.append(zh_vocab.decode(predicted_ids.tolist()))
                references.append(zh_vocab.decode(reference_ids.tolist()))
            if process is not None:
                peak_memory_mb = max(peak_memory_mb or 0.0, process.memory_info().rss / (1024 * 1024))
    elapsed = time.perf_counter() - start

    return {
        "bleu4": compute_bleu4(hypotheses, references),
        "rouge_l": compute_rouge_l(hypotheses, references),
        "wer": compute_wer(hypotheses, references),
        "latency_ms": elapsed / max(1, len(hypotheses)) * 1000.0,
        "peak_memory_mb": peak_memory_mb,
        "status": "ok",
    }


def evaluate_onnx_model(onnx_dir: Path, config_path: Path, test_pairs: List[tuple[str, str]]) -> Dict[str, object]:
    pipeline = TranslationPipeline(model_dir=onnx_dir.as_posix(), config_path=config_path.as_posix())
    process = _process()
    peak_memory_mb = process.memory_info().rss / (1024 * 1024) if process is not None else None
    hypotheses: List[str] = []
    references: List[str] = []

    start = time.perf_counter()
    for gloss_text, reference_text in test_pairs:
        hypotheses.append(pipeline.translate(gloss_text))
        references.append(reference_text)
        if process is not None:
            peak_memory_mb = max(peak_memory_mb or 0.0, process.memory_info().rss / (1024 * 1024))
    elapsed = time.perf_counter() - start

    return {
        "bleu4": compute_bleu4(hypotheses, references),
        "rouge_l": compute_rouge_l(hypotheses, references),
        "wer": compute_wer(hypotheses, references),
        "latency_ms": elapsed / max(1, len(hypotheses)) * 1000.0,
        "peak_memory_mb": peak_memory_mb,
        "status": "ok",
    }


def default_checkpoint_path(checkpoint_dir: Path, filename: str) -> Path:
    return checkpoint_dir / filename


def safe_evaluate_pytorch(label: str, checkpoint_path: Path, checkpoint_dir: Path, config: Dict, data_path: Path) -> Dict[str, object]:
    if not checkpoint_path.exists():
        return {
            "model": label,
            "checkpoint": checkpoint_path.name,
            "bleu4": None,
            "rouge_l": None,
            "wer": None,
            "latency_ms": None,
            "peak_memory_mb": None,
            "status": "missing",
        }
    try:
        resolved_checkpoint_dir = checkpoint_path.parent if checkpoint_path.parent.exists() else checkpoint_dir
        resolved_config = load_config(resolved_checkpoint_dir, None) if (resolved_checkpoint_dir / "runtime_config.yaml").exists() else config
        metrics = evaluate_pytorch_model(checkpoint_path, resolved_checkpoint_dir, resolved_config, data_path)
        metrics["model"] = label
        metrics["checkpoint"] = checkpoint_path.name
        return metrics
    except Exception as exc:  # pragma: no cover - runtime integration path
        return {
            "model": label,
            "checkpoint": checkpoint_path.name,
            "bleu4": None,
            "rouge_l": None,
            "wer": None,
            "latency_ms": None,
            "peak_memory_mb": None,
            "status": f"error: {exc}",
        }


def safe_evaluate_onnx(label: str, onnx_dir: Path, config_path: Path, test_pairs: List[tuple[str, str]]) -> Dict[str, object]:
    encoder_path = onnx_dir / "encoder.onnx"
    decoder_path = onnx_dir / "decoder.onnx"
    encoder_int8_path = onnx_dir / "encoder.int8.onnx"
    decoder_int8_path = onnx_dir / "decoder.int8.onnx"
    if not ((encoder_path.exists() and decoder_path.exists()) or (encoder_int8_path.exists() and decoder_int8_path.exists())):
        return {
            "model": label,
            "checkpoint": onnx_dir.as_posix(),
            "bleu4": None,
            "rouge_l": None,
            "wer": None,
            "latency_ms": None,
            "peak_memory_mb": None,
            "status": "missing",
        }
    try:
        metrics = evaluate_onnx_model(onnx_dir, config_path, test_pairs)
        metrics["model"] = label
        metrics["checkpoint"] = onnx_dir.as_posix()
        return metrics
    except Exception as exc:  # pragma: no cover - runtime integration path
        return {
            "model": label,
            "checkpoint": onnx_dir.as_posix(),
            "bleu4": None,
            "rouge_l": None,
            "wer": None,
            "latency_ms": None,
            "peak_memory_mb": None,
            "status": f"error: {exc}",
        }


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def print_markdown_table(rows: List[Dict[str, object]]) -> None:
    headers = ["Model", "Checkpoint", "BLEU-4", "ROUGE-L", "WER", "Latency(ms)", "PeakMemory(MB)", "Status"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| {model} | {checkpoint} | {bleu4} | {rouge_l} | {wer} | {latency_ms} | {peak_memory_mb} | {status} |".format(
                model=row["model"],
                checkpoint=row["checkpoint"],
                bleu4=format_value(row.get("bleu4")),
                rouge_l=format_value(row.get("rouge_l")),
                wer=format_value(row.get("wer")),
                latency_ms=format_value(row.get("latency_ms")),
                peak_memory_mb=format_value(row.get("peak_memory_mb")),
                status=row["status"],
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一评估不同版本的 Gloss -> 中文 模型")
    parser.add_argument("--checkpoint-dir", type=str, default=(PROJECT_ROOT / "checkpoints").as_posix())
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--enhanced", type=str, default=None)
    parser.add_argument("--noise", type=str, default=None)
    parser.add_argument("--distilled", type=str, default=None)
    parser.add_argument("--onnx-dir", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    config_path = Path(args.config) if args.config else None
    config = load_config(checkpoint_dir, config_path)
    data_dir = PROJECT_ROOT / "datasets"
    test_data_path = Path(args.test_data) if args.test_data else resolve_dataset_path(data_dir, "test.tsv", "test.csv")
    resolved_config_path = config_path or (checkpoint_dir / "runtime_config.yaml" if (checkpoint_dir / "runtime_config.yaml").exists() else PROJECT_ROOT / "configs" / "default.yaml")
    test_pairs = read_parallel_pairs(test_data_path.as_posix())

    rows = [
        safe_evaluate_pytorch(
            "原始 BiGRU baseline",
            Path(args.baseline) if args.baseline else default_checkpoint_path(checkpoint_dir, "baseline_model.pt"),
            checkpoint_dir,
            config,
            test_data_path,
        ),
        safe_evaluate_pytorch(
            "增强编码器版本",
            Path(args.enhanced) if args.enhanced else default_checkpoint_path(checkpoint_dir, "best_model.pt"),
            checkpoint_dir,
            config,
            test_data_path,
        ),
        safe_evaluate_pytorch(
            "增强编码器 + noise augmentation",
            Path(args.noise) if args.noise else default_checkpoint_path(checkpoint_dir, "noise_aug_model.pt"),
            checkpoint_dir,
            config,
            test_data_path,
        ),
        safe_evaluate_pytorch(
            "蒸馏后的最终模型",
            Path(args.distilled) if args.distilled else default_checkpoint_path(checkpoint_dir, "distilled_model.pt"),
            checkpoint_dir,
            config,
            test_data_path,
        ),
    ]

    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else checkpoint_dir
    rows.append(safe_evaluate_onnx("ONNX / INT8", onnx_dir, resolved_config_path, test_pairs))
    print_markdown_table(rows)


if __name__ == "__main__":
    main()


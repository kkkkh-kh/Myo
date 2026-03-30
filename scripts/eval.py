"""scripts/eval.py

Evaluate a trained Seq2Seq checkpoint on train/val/test splits and export predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from data.dataset import GlossChineseDataset
from data.vocabulary import Vocabulary
from model.decoder import ChineseDecoder
from model.encoder import GlossEncoder
from model.seq2seq import Seq2Seq
from train.checkpointing import load_checkpoint_into_model
from train.trainer import Trainer


def _path_candidates(raw_path: str, base_dir: Path) -> list[Path]:
    raw = Path(raw_path)
    candidates: list[Path] = [raw]
    if not raw.is_absolute():
        candidates.append(base_dir / raw)

    if raw.parts and raw.parts[0].lower() == "myo" and len(raw.parts) > 1:
        stripped = Path(*raw.parts[1:])
        candidates.append(base_dir / stripped)
        candidates.append(base_dir.parent / stripped)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_existing_path(raw_path: str, base_dir: Path) -> Path:
    for candidate in _path_candidates(raw_path, base_dir):
        if candidate.exists():
            return candidate.resolve()
    candidates = ", ".join(path.as_posix() for path in _path_candidates(raw_path, base_dir))
    raise FileNotFoundError(f"Path not found: {raw_path}. Tried: {candidates}")


def resolve_optional_path(raw_path: Optional[str], base_dir: Path) -> Optional[Path]:
    if not raw_path:
        return None
    return resolve_existing_path(raw_path, base_dir)


def resolve_output_path(raw_path: str, base_dir: Path) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0].lower() == "myo" and len(raw.parts) > 1:
        return (base_dir / Path(*raw.parts[1:])).resolve()
    return (base_dir / raw).resolve()


def resolve_dataset_path(data_dir: Path, *candidates: str) -> Path:
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path.resolve()
    tried = ", ".join((data_dir / name).as_posix() for name in candidates)
    raise FileNotFoundError(f"No dataset file found. Tried: {tried}")


def resolve_train_path(data_dir: Path, config: Dict, train_file: Optional[str]) -> Path:
    if train_file:
        return resolve_existing_path(train_file, data_dir)

    augment_cfg = config.get("word_order_augment", {})
    if augment_cfg.get("enabled", False):
        for candidate in ("train_augmented.tsv", "train_augmented.csv"):
            path = data_dir / candidate
            if path.exists():
                return path.resolve()

    return resolve_dataset_path(data_dir, "train.tsv", "train.csv")


def resolve_split_path(eval_split: str, data_dir: Path, config: Dict, args: argparse.Namespace) -> Path:
    if eval_split == "train":
        return resolve_train_path(data_dir, config, args.train_file)
    if eval_split == "val":
        if args.val_file:
            return resolve_existing_path(args.val_file, data_dir)
        return resolve_dataset_path(data_dir, "val.tsv", "dev.tsv", "val.csv", "dev.csv")
    if eval_split == "test":
        if args.test_file:
            return resolve_existing_path(args.test_file, data_dir)
        return resolve_dataset_path(data_dir, "test.tsv", "test.csv")
    raise ValueError(f"Unsupported eval split: {eval_split}")


def load_config(config_path: Optional[Path], checkpoint_dir: Path) -> tuple[Dict, Path]:
    if config_path is not None:
        target = config_path
    else:
        runtime_cfg = checkpoint_dir / "runtime_config.yaml"
        target = runtime_cfg if runtime_cfg.exists() else PROJECT_ROOT / "configs" / "default.yaml"

    with target.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file), target.resolve()


def build_model_and_vocab(config: Dict, checkpoint_dir: Path) -> tuple[Seq2Seq, Vocabulary, Vocabulary]:
    gloss_vocab = Vocabulary.load(checkpoint_dir / "gloss_vocab.json")
    zh_vocab = Vocabulary.load(checkpoint_dir / "zh_vocab.json")

    config.setdefault("model", {})
    config.setdefault("encoder", {})
    config["model"]["gloss_vocab_size"] = len(gloss_vocab)
    config["model"]["zh_vocab_size"] = len(zh_vocab)

    encoder_cfg = config.get("encoder", {})
    model_cfg = config.get("model", {})

    encoder = GlossEncoder(
        gloss_vocab_size=config["model"]["gloss_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        pad_id=Vocabulary.PAD_ID,
        use_sen=bool(encoder_cfg.get("use_sen", False)),
        sen_reduction=int(encoder_cfg.get("sen_reduction", 16)),
        use_transformer=bool(encoder_cfg.get("use_transformer", False)),
        transformer_layers=int(encoder_cfg.get("transformer_layers", 2)),
        transformer_heads=int(encoder_cfg.get("transformer_heads", 4)),
        transformer_dropout=float(encoder_cfg.get("transformer_dropout", 0.1)),
    )
    decoder = ChineseDecoder(
        zh_vocab_size=config["model"]["zh_vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        use_word_order_attention=bool(model_cfg.get("use_word_order_attention", False)),
        max_relative_position=int(model_cfg.get("max_relative_position", 64)),
        use_order_guidance=bool(model_cfg.get("use_order_guidance", True)),
        guidance_lambda_init=float(model_cfg.get("guidance_lambda_init", 1.0)),
        guidance_decay_ratio=float(model_cfg.get("guidance_decay_ratio", 0.3)),
    )
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    return model, gloss_vocab, zh_vocab


def build_dataloader(
    data_path: Path,
    gloss_vocab: Vocabulary,
    zh_vocab: Vocabulary,
    config: Dict,
    batch_size: int,
) -> DataLoader:
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=GlossChineseDataset.collate_fn,
    )


def normalize_eval_split(raw_split: str) -> str:
    normalized = raw_split.strip().lower()
    alias_map = {
        "validation": "val",
        "dev": "val",
    }
    return alias_map.get(normalized, normalized)


def write_jsonl(path: Path, rows: list[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on train/val/test split.")
    parser.add_argument("--config", type=str, default=None, help="Path to runtime_config.yaml.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt checkpoint.")
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["train", "val", "test", "validation", "dev"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--data_dir", type=str, default=(PROJECT_ROOT / "datasets").as_posix(), help="Dataset directory.")
    parser.add_argument("--train_file", type=str, default=None, help="Override train file path.")
    parser.add_argument("--val_file", type=str, default=None, help="Override val/dev file path.")
    parser.add_argument("--test_file", type=str, default=None, help="Override test file path.")
    parser.add_argument("--output", type=str, default=None, help="Output jsonl path. Default: <checkpoint_dir>/<split>_samples.jsonl")
    parser.add_argument("--metrics_output", type=str, default=None, help="Metrics json path. Default: <checkpoint_dir>/<split>_metrics.json")
    parser.add_argument("--sample_size", type=int, default=None, help="Export sample count. Use negative value with --export_all to export all rows.")
    parser.add_argument("--export_all", action="store_true", help="Export full split predictions instead of sampled rows.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override for evaluation.")
    parser.add_argument("--beam_size", type=int, default=None, help="Beam size override (default keeps training config).")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu/cuda/cuda:0.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    checkpoint_path = resolve_existing_path(args.checkpoint, PROJECT_ROOT)
    checkpoint_dir = checkpoint_path.parent
    config_path = resolve_optional_path(args.config, PROJECT_ROOT)
    config, resolved_config_path = load_config(config_path, checkpoint_dir)

    config.setdefault("train", {})
    config.setdefault("model", {})
    config.setdefault("data", {})

    if args.device:
        config["device"] = args.device
    elif "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    if args.beam_size is not None:
        config["train"]["validation_beam_size"] = max(1, int(args.beam_size))

    if "validation_beam_size" not in config["train"]:
        config["train"]["validation_beam_size"] = int(config.get("deploy", {}).get("beam_size", 1))

    if "validation_sample_size" not in config["train"]:
        config["train"]["validation_sample_size"] = 5

    data_dir = resolve_existing_path(args.data_dir, PROJECT_ROOT)
    eval_split = normalize_eval_split(args.eval_split)
    split_path = resolve_split_path(eval_split, data_dir, config, args)

    model, gloss_vocab, zh_vocab = build_model_and_vocab(config, checkpoint_dir)
    batch_size = int(args.batch_size or config["train"].get("batch_size", 64))
    dataloader = build_dataloader(split_path, gloss_vocab, zh_vocab, config, batch_size)

    trainer = Trainer(model=model, optimizer=None, scheduler=None, config=config)
    checkpoint_obj = load_checkpoint_into_model(model, checkpoint_path, map_location="cpu")
    trainer.current_epoch = int(checkpoint_obj.get("epoch", 0)) if isinstance(checkpoint_obj, dict) else 0

    export_all = bool(args.export_all)
    sample_size = args.sample_size
    if sample_size is not None and sample_size < 0:
        export_all = True
        sample_size = None

    evaluation = trainer.evaluate_split(
        dataloader=dataloader,
        split=eval_split,
        sample_size=sample_size,
        collect_predictions=export_all,
    )

    default_samples_path = checkpoint_dir / f"{eval_split}_samples.jsonl"
    default_metrics_path = checkpoint_dir / f"{eval_split}_metrics.json"
    samples_path = resolve_output_path(args.output, PROJECT_ROOT) if args.output else default_samples_path
    metrics_path = resolve_output_path(args.metrics_output, PROJECT_ROOT) if args.metrics_output else default_metrics_path
    if args.output:
        samples_path.parent.mkdir(parents=True, exist_ok=True)
    if args.metrics_output:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    rows = evaluation.get("predictions", []) if export_all else evaluation.get("samples", [])
    write_jsonl(samples_path, rows)

    metrics_payload = {
        "split": eval_split,
        "checkpoint": checkpoint_path.as_posix(),
        "config": resolved_config_path.as_posix(),
        "data_file": split_path.as_posix(),
        "export_all": export_all,
        "exported_rows": len(rows),
        "dataset_size": len(dataloader.dataset),
        "loss": float(evaluation.get("loss", 0.0)),
        "bleu4": float(evaluation.get("bleu4", 0.0)),
        "rouge_l": float(evaluation.get("rouge_l", 0.0)),
        "wer": float(evaluation.get("wer", 0.0)),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, ensure_ascii=False, indent=2)

    print(f"Eval split      : {eval_split}")
    print(f"Data file       : {split_path}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Samples output  : {samples_path}")
    print(f"Metrics output  : {metrics_path}")
    print(
        "Metrics         : loss={loss:.4f}, BLEU-4={bleu4:.2f}, ROUGE-L={rouge_l:.2f}, WER={wer:.2f}".format(
            **metrics_payload
        )
    )


if __name__ == "__main__":
    main()

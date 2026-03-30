# Module description: Offline dataset augmentation entrypoint for word-order robustness.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from data.preprocess import read_parallel_pairs
from datasets.word_order_augment import DEFAULT_SYNONYM_DICT, SIGN_LANGUAGE_ORDER_RULES, WordOrderAugmentor


def _load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _write_tsv(samples: List[Tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for gloss, zh in samples:
            file.write(f"{gloss}\t{zh}\n")


def main(args: argparse.Namespace) -> None:
    """Run offline augmentation and persist enhanced training pairs."""
    config = _load_config(Path(args.config))
    augment_cfg = config.get("word_order_augment", {})
    strategies = args.strategies or augment_cfg.get(
        "strategies",
        ["temporal_shift", "negation_shift", "subsequence_sampling", "backtrans_sim", "synonym_replace"],
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats) if args.stats else output_path.parent / "augment_stats.json"
    samples = read_parallel_pairs(input_path.as_posix())

    synonym_dict = dict(DEFAULT_SYNONYM_DICT)
    synonym_dict.update(augment_cfg.get("synonym_dict", {}))

    augmentor = WordOrderAugmentor(
        gloss_vocab=None,
        zh_vocab=None,
        time_tokens=augment_cfg.get("time_tokens", SIGN_LANGUAGE_ORDER_RULES["time_tokens"]),
        neg_tokens=augment_cfg.get("neg_tokens", SIGN_LANGUAGE_ORDER_RULES["neg_tokens"]),
        synonym_dict=synonym_dict,
        strategies=strategies,
        augment_ratio=float(args.augment_ratio),
        seed=int(args.seed),
    )

    augmented_samples = augmentor.augment_dataset(samples)
    _write_tsv(augmented_samples, output_path)

    stats = dict(augmentor.last_stats)
    stats["input_path"] = input_path.as_posix()
    stats["output_path"] = output_path.as_posix()
    stats["strategies"] = list(strategies)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    print(f"原始样本数: {stats.get('original_count', 0)}")
    print(f"增强后样本数: {stats.get('augmented_count', 0)}")
    print(f"输出文件: {output_path}")
    print(f"统计文件: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线生成语序增强训练集。")
    parser.add_argument("--input", default="datasets/train.tsv", help="输入训练文件（TSV/CSV）")
    parser.add_argument("--output", default="datasets/train_augmented.tsv", help="增强后输出 TSV")
    parser.add_argument("--stats", default="", help="增强统计 JSON 输出路径")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--augment_ratio", type=float, default=4.0, help="增强倍率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="增强策略列表，例如 temporal_shift negation_shift backtrans_sim",
    )
    main(parser.parse_args())


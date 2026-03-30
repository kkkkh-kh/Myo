# Module description: Dedicated evaluation script for word-order quality in gloss-to-Chinese translation.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

from data.preprocess import read_parallel_pairs
from train.evaluate import compute_bleu4, compute_rouge_l, compute_wer


MODEL_VARIANTS = [
    ("基线（无增强）", ["baseline_predictions.txt"]),
    ("+数据增强", ["data_augment_predictions.txt"]),
    ("+语序注意力", ["order_attention_predictions.txt"]),
    ("+语序损失", ["order_loss_predictions.txt"]),
    ("+后处理", ["postprocess_predictions.txt"]),
    ("完整方案", ["full_predictions.txt", "val_predictions.txt"]),
]

TIME_WORDS = {
    "昨天",
    "今天",
    "明天",
    "后天",
    "前天",
    "上午",
    "下午",
    "晚上",
    "早上",
    "中午",
    "刚才",
    "现在",
    "以前",
    "以后",
}
NEG_WORDS = {"不", "没", "没有", "别", "不要", "未", "勿"}
SUBJECT_WORDS = {"我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "大家", "学生", "老师", "残疾人"}
VERB_HINTS = {"买", "去", "看", "说", "申请", "提交", "学习", "工作", "办理", "补偿", "补助", "拿", "给", "做"}


def tokenize_zh(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if " " in text:
        return [token for token in text.split() if token]
    if jieba is not None:
        return [token for token in jieba.lcut(text, cut_all=False) if token.strip()]
    return [char for char in text if char.strip()]


def kendall_tau(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    common = [token for token in dict.fromkeys(ref_tokens) if token in set(pred_tokens)]
    if len(common) < 2:
        return 0.0

    pred_pos = {token: pred_tokens.index(token) for token in common}
    ref_pos = {token: ref_tokens.index(token) for token in common}

    concordant = 0
    discordant = 0
    total_pairs = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            left, right = common[i], common[j]
            pred_order = pred_pos[left] - pred_pos[right]
            ref_order = ref_pos[left] - ref_pos[right]
            if pred_order == 0 or ref_order == 0:
                continue
            total_pairs += 1
            if pred_order * ref_order > 0:
                concordant += 1
            else:
                discordant += 1

    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs


def find_first_verb(tokens: Sequence[str]) -> int:
    for index, token in enumerate(tokens):
        if token in VERB_HINTS or token.endswith(("买", "去", "看", "办", "做", "学", "说")):
            return index
    return -1


def component_position_match(pred_tokens: Sequence[str], ref_tokens: Sequence[str], component_words: set[str]) -> bool:
    pred_idx = next((i for i, t in enumerate(pred_tokens) if t in component_words), -1)
    ref_idx = next((i for i, t in enumerate(ref_tokens) if t in component_words), -1)
    if pred_idx < 0 or ref_idx < 0:
        return False

    pred_verb = find_first_verb(pred_tokens)
    ref_verb = find_first_verb(ref_tokens)
    if pred_verb < 0 or ref_verb < 0:
        return pred_idx == ref_idx
    return (pred_idx < pred_verb) == (ref_idx < ref_verb)


def evaluate_word_order(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    pair_count = max(1, min(len(hypotheses), len(references)))
    hypotheses = hypotheses[:pair_count]
    references = references[:pair_count]

    bleu4 = compute_bleu4(hypotheses, references)
    rouge_l = compute_rouge_l(hypotheses, references)
    wer = compute_wer(hypotheses, references)

    taus: List[float] = []
    time_hits = 0
    neg_hits = 0
    subject_hits = 0
    object_hits = 0
    time_total = 0
    neg_total = 0
    subject_total = 0
    object_total = 0

    for hypothesis, reference in zip(hypotheses, references):
        hyp_tokens = tokenize_zh(hypothesis)
        ref_tokens = tokenize_zh(reference)
        taus.append(kendall_tau(hyp_tokens, ref_tokens))

        if any(token in TIME_WORDS for token in ref_tokens):
            time_total += 1
            if component_position_match(hyp_tokens, ref_tokens, TIME_WORDS):
                time_hits += 1

        if any(token in NEG_WORDS for token in ref_tokens):
            neg_total += 1
            if component_position_match(hyp_tokens, ref_tokens, NEG_WORDS):
                neg_hits += 1

        ref_subject = next((token for token in ref_tokens if token in SUBJECT_WORDS), None)
        if ref_subject is not None:
            subject_total += 1
            hyp_subject = next((token for token in hyp_tokens if token in SUBJECT_WORDS), None)
            if hyp_subject == ref_subject:
                subject_hits += 1

        ref_verb = find_first_verb(ref_tokens)
        if ref_verb >= 0 and ref_verb < len(ref_tokens) - 1:
            object_total += 1
            ref_object = ref_tokens[-1]
            hyp_object = hyp_tokens[-1] if hyp_tokens else ""
            if hyp_object == ref_object:
                object_hits += 1

    def safe_ratio(hit: int, total: int) -> float:
        return 100.0 * hit / total if total > 0 else 0.0

    return {
        "bleu4": bleu4,
        "rouge_l": rouge_l,
        "wer": wer,
        "order_tau": sum(taus) / len(taus) if taus else 0.0,
        "time_acc": safe_ratio(time_hits, time_total),
        "neg_acc": safe_ratio(neg_hits, neg_total),
        "subject_acc": safe_ratio(subject_hits, subject_total),
        "object_acc": safe_ratio(object_hits, object_total),
    }


def load_predictions(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def find_variant_file(checkpoints_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        candidate = checkpoints_dir / name
        if candidate.exists():
            return candidate
    return None


def build_report(rows: List[Dict[str, str]], output_path: Path) -> None:
    header = [
        "| 模型版本 | BLEU-4 | ROUGE-L | WER | 语序τ | 时间状语% | 否定词% | 主语% | 宾语% |",
        "|---------|--------|---------|-----|------|---------|--------|------|------|",
    ]
    table_rows = [
        f"| {row['name']} | {row['bleu4']} | {row['rouge_l']} | {row['wer']} | {row['order_tau']} | {row['time_acc']} | {row['neg_acc']} | {row['subject_acc']} | {row['object_acc']} |"
        for row in rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(header + table_rows) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    checkpoints_dir = Path(args.checkpoints_dir)
    output_report = Path(args.output_report)

    references = [zh for _, zh in read_parallel_pairs(args.test_file)]
    rows: List[Dict[str, str]] = []

    for variant_name, candidate_files in MODEL_VARIANTS:
        prediction_file = find_variant_file(checkpoints_dir, candidate_files)
        if not prediction_file:
            rows.append(
                {
                    "name": variant_name,
                    "bleu4": "-",
                    "rouge_l": "-",
                    "wer": "-",
                    "order_tau": "-",
                    "time_acc": "-",
                    "neg_acc": "-",
                    "subject_acc": "-",
                    "object_acc": "-",
                }
            )
            continue

        hypotheses = load_predictions(prediction_file)
        metrics = evaluate_word_order(hypotheses, references)
        rows.append(
            {
                "name": variant_name,
                "bleu4": f"{metrics['bleu4']:.2f}",
                "rouge_l": f"{metrics['rouge_l']:.2f}",
                "wer": f"{metrics['wer']:.2f}",
                "order_tau": f"{metrics['order_tau']:.3f}",
                "time_acc": f"{metrics['time_acc']:.1f}",
                "neg_acc": f"{metrics['neg_acc']:.1f}",
                "subject_acc": f"{metrics['subject_acc']:.1f}",
                "object_acc": f"{metrics['object_acc']:.1f}",
            }
        )

    build_report(rows, output_report)
    print(f"评估完成，报告已写入: {output_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="语序质量专项评估")
    parser.add_argument("--test_file", default="datasets/test.tsv", help="测试集文件路径")
    parser.add_argument("--checkpoints_dir", default="./checkpoints", help="预测文件所在目录")
    parser.add_argument("--output_report", default="./checkpoints/word_order_report.md", help="Markdown 报告输出路径")
    main(parser.parse_args())

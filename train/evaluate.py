from typing import Dict, Iterable, List

import sacrebleu
import torch
from rouge_score import rouge_scorer

from data.vocabulary import Vocabulary


def _space_for_metrics(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if " " in stripped:
        return stripped
    return " ".join(list(stripped))


def compute_bleu4(hypotheses: List[str], references: List[str]) -> float:
    if not hypotheses:
        return 0.0
    bleu = sacrebleu.corpus_bleu(
        [_space_for_metrics(item) for item in hypotheses],
        [[_space_for_metrics(item) for item in references]],
        force=True,
        lowercase=False,
        tokenize="none",
    )
    return float(bleu.score)


def compute_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    if not hypotheses:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(reference, hypothesis)["rougeL"].fmeasure
        for hypothesis, reference in zip(hypotheses, references)
    ]
    return float(sum(scores) / len(scores) * 100.0)


def _edit_distance(source: List[str], target: List[str]) -> int:
    rows = len(source) + 1
    cols = len(target) + 1
    dp = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col
    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if source[row - 1] == target[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + substitution_cost,
            )
    return dp[-1][-1]


def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    total_distance = 0
    total_tokens = 0
    for hypothesis, reference in zip(hypotheses, references):
        hyp_tokens = hypothesis.split() if " " in hypothesis else list(hypothesis)
        ref_tokens = reference.split() if " " in reference else list(reference)
        total_distance += _edit_distance(hyp_tokens, ref_tokens)
        total_tokens += max(1, len(ref_tokens))
    return float(total_distance / max(1, total_tokens) * 100.0)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable,
    gloss_vocab: Vocabulary,
    zh_vocab: Vocabulary,
) -> Dict[str, float]:
    del gloss_vocab
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    hypotheses: List[str] = []
    references: List[str] = []
    with torch.no_grad():
        for gloss_ids, _, zh_ids, _ in dataloader:
            gloss_ids = gloss_ids.to(device)
            predictions = model.translate(gloss_ids, max_len=zh_ids.size(1) - 1)
            for predicted_ids, reference_ids in zip(predictions.cpu(), zh_ids):
                hypotheses.append(zh_vocab.decode(predicted_ids.tolist()))
                references.append(zh_vocab.decode(reference_ids.tolist()))

    if was_training:
        model.train()
    return {
        "bleu4": compute_bleu4(hypotheses, references),
        "rouge_l": compute_rouge_l(hypotheses, references),
        "wer": compute_wer(hypotheses, references),
    }

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from data.preprocess import clean_chinese_text, clean_gloss_text, read_parallel_pairs, tokenize_chinese


DEFAULT_SOURCE = "real"
DEFAULT_TASK_PREFIX = "translate gloss to chinese:"


@dataclass(frozen=True)
class ParallelExample:
    gloss: str
    text: str
    source: str = DEFAULT_SOURCE


def _sanitize_source_name(source_name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", (source_name or DEFAULT_SOURCE).strip().lower()).strip("_")
    return sanitized or DEFAULT_SOURCE


def source_special_token(source_name: str) -> str:
    return f"<src_{_sanitize_source_name(source_name)}>"


def format_gloss_prompt(
    gloss: str,
    *,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    source: Optional[str] = None,
    include_source_tag: bool = False,
) -> str:
    normalized_gloss = clean_gloss_text(gloss)
    prefix = (task_prefix or "").strip()
    pieces: List[str] = []
    if prefix:
        pieces.append(prefix)
    if include_source_tag and source:
        pieces.append(source_special_token(source))
    if normalized_gloss:
        pieces.append(normalized_gloss)
    return " ".join(piece for piece in pieces if piece).strip()


def _build_example(gloss: str, text: str, source: str) -> ParallelExample:
    normalized_gloss = clean_gloss_text(gloss)
    normalized_text = clean_chinese_text(text)
    if not normalized_gloss or not normalized_text:
        raise ValueError("Gloss and Chinese text must both be non-empty after normalization.")
    return ParallelExample(gloss=normalized_gloss, text=normalized_text, source=_sanitize_source_name(source))


def _resolve_json_value(record: Dict[str, object], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        value = record.get(candidate)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def read_parallel_examples(path: str, *, default_source: Optional[str] = None) -> List[ParallelExample]:
    data_path = Path(path)
    source_name = _sanitize_source_name(default_source or data_path.stem)
    if data_path.suffix.lower() != ".jsonl":
        return [
            _build_example(gloss, chinese, source_name)
            for gloss, chinese in read_parallel_pairs(data_path.as_posix())
        ]

    examples: List[ParallelExample] = []
    with data_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            gloss = _resolve_json_value(record, ["gloss", "source_gloss", "input"])
            text = _resolve_json_value(record, ["chinese", "text", "target", "translation", "reference"])
            source = _resolve_json_value(record, ["source", "provenance"]) or source_name
            if not gloss or not text:
                raise ValueError(f"Invalid JSONL record on line {line_number}: {record}")
            examples.append(_build_example(gloss, text, source))
    return examples


def collect_source_tokens(examples: Iterable[ParallelExample]) -> List[str]:
    return sorted({source_special_token(example.source) for example in examples})


def mix_parallel_examples(
    real_examples: Sequence[ParallelExample],
    synthetic_groups: Optional[Dict[str, Sequence[ParallelExample]]] = None,
    *,
    max_synthetic_ratio: float = 1.0,
    seed: int = 42,
) -> List[ParallelExample]:
    mixed = list(real_examples)
    if not synthetic_groups or max_synthetic_ratio <= 0.0:
        return mixed

    real_count = len(real_examples)
    max_synthetic_examples = int(real_count * max_synthetic_ratio)
    if max_synthetic_examples <= 0:
        return mixed

    rng = random.Random(seed)
    shuffled_buckets: Dict[str, List[ParallelExample]] = {}
    for source_name, examples in synthetic_groups.items():
        bucket = list(examples)
        rng.shuffle(bucket)
        shuffled_buckets[source_name] = bucket

    source_names = sorted(shuffled_buckets)
    synthetic_examples: List[ParallelExample] = []
    while len(synthetic_examples) < max_synthetic_examples:
        appended = False
        for source_name in source_names:
            bucket = shuffled_buckets[source_name]
            if not bucket:
                continue
            synthetic_examples.append(bucket.pop())
            appended = True
            if len(synthetic_examples) >= max_synthetic_examples:
                break
        if not appended:
            break

    mixed.extend(synthetic_examples)
    rng.shuffle(mixed)
    return mixed


def load_training_examples(
    train_path: str,
    *,
    synthetic_paths: Optional[Sequence[str]] = None,
    max_synthetic_ratio: float = 1.0,
    seed: int = 42,
) -> List[ParallelExample]:
    real_examples = read_parallel_examples(train_path, default_source=DEFAULT_SOURCE)
    if not synthetic_paths:
        return real_examples

    grouped: Dict[str, List[ParallelExample]] = defaultdict(list)
    for synthetic_path in synthetic_paths:
        path = Path(synthetic_path)
        source_name = _sanitize_source_name(path.stem)
        grouped[source_name].extend(read_parallel_examples(path.as_posix(), default_source=source_name))
    return mix_parallel_examples(real_examples, grouped, max_synthetic_ratio=max_synthetic_ratio, seed=seed)


def _surface_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    ratio = SequenceMatcher(a=left, b=right).ratio()
    left_set = set(left)
    right_set = set(right)
    overlap = len(left_set & right_set) / max(1, len(left_set | right_set))
    return max(ratio, overlap)


def build_semantic_neighbor_map(
    texts: Sequence[str],
    encode_surface_token: Callable[[str], Sequence[int]],
    *,
    zh_tokenizer_mode: str = "jieba",
    top_k: int = 4,
    min_similarity: float = 0.5,
) -> Dict[int, List[Tuple[int, float]]]:
    token_counter: Counter = Counter()
    for text in texts:
        token_counter.update(tokenize_chinese(text, mode=zh_tokenizer_mode))

    surface_by_id: Dict[int, Tuple[str, int]] = {}
    for token, count in token_counter.items():
        piece_ids = [int(piece_id) for piece_id in encode_surface_token(token)]
        if len(piece_ids) != 1:
            continue
        token_id = piece_ids[0]
        previous = surface_by_id.get(token_id)
        if previous is None or count > previous[1]:
            surface_by_id[token_id] = (token, count)

    token_items = sorted(surface_by_id.items())
    neighbor_map: Dict[int, List[Tuple[int, float]]] = {}
    for token_id, (token, _) in token_items:
        candidates: List[Tuple[int, float]] = []
        for other_id, (other_token, _) in token_items:
            if other_id == token_id:
                continue
            similarity = _surface_similarity(token, other_token)
            if similarity >= min_similarity:
                candidates.append((other_id, similarity))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if top_k > 0:
            candidates = candidates[:top_k]
        if not candidates:
            continue
        total_similarity = sum(score for _, score in candidates)
        if total_similarity <= 0.0:
            continue
        neighbor_map[token_id] = [
            (neighbor_id, similarity / total_similarity)
            for neighbor_id, similarity in candidates
        ]
    return neighbor_map


class ApproxSemanticAwareLabelSmoother:
    def __init__(
        self,
        neighbor_map: Optional[Dict[int, List[Tuple[int, float]]]] = None,
        *,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.neighbor_map = neighbor_map or {}
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError("logits must have shape [batch, seq_len, vocab_size]")
        if labels.dim() != 2:
            raise ValueError("labels must have shape [batch, seq_len]")

        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_labels = labels.reshape(-1)
        valid_mask = flat_labels.ne(self.ignore_index)
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        valid_logits = flat_logits[valid_mask]
        valid_labels = flat_labels[valid_mask]
        log_probs = F.log_softmax(valid_logits, dim=-1)

        losses: List[torch.Tensor] = []
        confidence = 1.0 - self.smoothing
        for row_index, label_id in enumerate(valid_labels.tolist()):
            row_log_probs = log_probs[row_index]
            neighbors = self.neighbor_map.get(int(label_id), []) if self.smoothing > 0.0 else []
            if neighbors:
                token_ids = [int(label_id)] + [neighbor_id for neighbor_id, _ in neighbors]
                weights = [confidence] + [self.smoothing * weight for _, weight in neighbors]
                index_tensor = torch.tensor(token_ids, dtype=torch.long, device=row_log_probs.device)
                weight_tensor = row_log_probs.new_tensor(weights)
                losses.append(-(row_log_probs.index_select(0, index_tensor) * weight_tensor).sum())
            else:
                losses.append(-row_log_probs[int(label_id)])
        return torch.stack(losses).mean()
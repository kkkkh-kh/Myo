# Module description: Word-order data augmentation utilities for low-resource gloss-to-Chinese training.

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import jieba
except ImportError:  # pragma: no cover - fallback only when jieba is unavailable.
    jieba = None


SIGN_LANGUAGE_ORDER_RULES = {
    "canonical": ["主语", "时间", "地点", "否定", "动词", "宾语"],
    "time_tokens": [
        "昨天",
        "今天",
        "明天",
        "上午",
        "下午",
        "晚上",
        "刚才",
        "以前",
        "以后",
        "现在",
        "最近",
    ],
    "neg_tokens": ["不", "没", "没有", "别", "不要", "未"],
    "location_tokens": ["学校", "医院", "家", "公司", "超市", "这里", "那里"],
}

DEFAULT_SYNONYM_DICT = {
    "买": "购买",
    "购买": "买",
    "去": "前往",
    "前往": "去",
    "看": "查看",
    "查看": "看",
    "申请": "提交",
    "提交": "申请",
    "补偿": "补助",
    "补助": "补偿",
    "学习": "读书",
    "读书": "学习",
    "帮助": "协助",
    "协助": "帮助",
}

DEFAULT_VERB_TOKENS = {
    "去",
    "来",
    "买",
    "卖",
    "吃",
    "喝",
    "看",
    "说",
    "申请",
    "提交",
    "学习",
    "工作",
    "办理",
    "补偿",
    "补助",
    "收到",
    "拿",
    "给",
    "找",
    "办",
    "取",
    "做",
}

SUBJECT_CANDIDATES = {"我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "大家", "残疾人", "学生", "老师"}

FUNCTION_WORDS = {"了", "着", "过", "在", "向", "对", "把", "被", "的", "地", "得", "吗", "呢", "啊"}

STRATEGY_ALIASES = {
    "component_swap": "component_swap",
    "temporal_shift": "temporal_shift",
    "negation_shift": "negation_shift",
    "backtrans_sim": "backtrans_sim",
    "subsequence": "subsequence_sampling",
    "subsequence_sampling": "subsequence_sampling",
    "synonym_replace": "synonym_replace",
}


class WordOrderAugmentor:
    """Generate word-order variants for low-resource gloss-to-Chinese datasets."""

    def __init__(
        self,
        gloss_vocab,
        zh_vocab,
        time_tokens: List[str],
        neg_tokens: List[str],
        synonym_dict: Dict[str, str],
        strategies: List[str] = None,
        augment_ratio: float = 3.0,
        seed: int = 42,
    ) -> None:
        """Initialize the augmentor.

        Args:
            gloss_vocab: Reserved for future vocabulary-aware augmentation.
            zh_vocab: Reserved for future vocabulary-aware augmentation.
            time_tokens: Time-related gloss tokens.
            neg_tokens: Negation-related gloss tokens.
            synonym_dict: Gloss synonym map.
            strategies: Enabled strategy names.
            augment_ratio: Requested dataset size multiplier.
            seed: Random seed for deterministic augmentation.
        """
        self.gloss_vocab = gloss_vocab
        self.zh_vocab = zh_vocab
        self.time_tokens = set(time_tokens or [])
        self.neg_tokens = set(neg_tokens or [])
        self.location_tokens = set(SIGN_LANGUAGE_ORDER_RULES["location_tokens"])
        self.synonym_dict = dict(DEFAULT_SYNONYM_DICT)
        self.synonym_dict.update(synonym_dict or {})
        self.strategies = [
            STRATEGY_ALIASES[name]
            for name in (strategies or ["temporal_shift", "negation_shift", "subsequence_sampling", "backtrans_sim"])
            if name in STRATEGY_ALIASES
        ]
        self.augment_ratio = max(1.0, float(augment_ratio))
        self.rng = random.Random(seed)
        self.verb_tokens = set(DEFAULT_VERB_TOKENS)
        self.last_stats: Dict[str, object] = {}

    def augment_dataset(self, samples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Augment a dataset and return deduplicated samples.

        Args:
            samples: List of ``(gloss_str, zh_sentence)`` pairs.

        Returns:
            Original plus augmented samples, deduplicated by pair content.
        """
        clean_samples = self._normalize_samples(samples)
        original_count = len(clean_samples)
        target_count = max(original_count, int(round(original_count * self.augment_ratio)))
        if original_count == 0:
            self.last_stats = {
                "original_count": 0,
                "target_count": 0,
                "augmented_count": 0,
                "strategy_counts": {},
                "order_distribution": {},
            }
            return []

        merged: List[Tuple[str, str]] = []
        seen = set()
        strategy_counts: Counter[str] = Counter()
        order_distribution: Counter[str] = Counter()
        candidates: List[Tuple[str, Tuple[str, str]]] = []

        for gloss, zh in clean_samples:
            key = (gloss, zh)
            if key not in seen:
                seen.add(key)
                merged.append(key)
                order_distribution[self._infer_order_pattern(gloss.split())] += 1

            gloss_tokens = gloss.split()
            for strategy_name in self.strategies:
                method = getattr(self, f"_{strategy_name}", None)
                if method is None:
                    continue
                for variant in method(gloss_tokens, zh):
                    candidates.append((strategy_name, variant))

        self.rng.shuffle(candidates)
        stagnation = 0
        while len(merged) < target_count and candidates:
            strategy_name, (new_gloss, new_zh) = candidates.pop()
            key = (self._normalize_space(new_gloss), self._normalize_space(new_zh))
            if not key[0] or not key[1] or key in seen:
                stagnation += 1
                if stagnation > 1000:
                    break
                continue
            seen.add(key)
            merged.append(key)
            strategy_counts[strategy_name] += 1
            order_distribution[self._infer_order_pattern(key[0].split())] += 1
            stagnation = 0

        self.last_stats = {
            "original_count": original_count,
            "target_count": target_count,
            "augmented_count": len(merged),
            "strategy_counts": dict(strategy_counts),
            "order_distribution": dict(order_distribution),
        }
        return merged

    def _normalize_samples(self, samples: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
        normalized: List[Tuple[str, str]] = []
        for gloss, zh in samples:
            gloss_clean = self._normalize_space(gloss)
            zh_clean = self._normalize_space(zh)
            if gloss_clean and zh_clean:
                normalized.append((gloss_clean, zh_clean))
        return normalized

    @staticmethod
    def _normalize_space(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _component_swap(self, gloss_tokens: List[str], zh_sentence: str) -> List[Tuple[str, str]]:
        """Swap likely subject and object components."""
        if len(gloss_tokens) < 3:
            return []
        verb_index = self._find_first_verb_index(gloss_tokens)
        if verb_index <= 0 or verb_index >= len(gloss_tokens) - 1:
            verb_index = 1
        subject = gloss_tokens[:1]
        middle = gloss_tokens[1 : verb_index + 1]
        obj = gloss_tokens[verb_index + 1 :]
        if not obj:
            return []
        swapped = obj + subject + middle
        if swapped == gloss_tokens:
            return []
        return [(" ".join(swapped), zh_sentence)]

    def _temporal_shift(self, gloss_tokens: List[str], zh_sentence: str) -> List[Tuple[str, str]]:
        """Shift time adverbials to front or tail positions."""
        time_indices = [index for index, token in enumerate(gloss_tokens) if token in self.time_tokens]
        if not time_indices:
            return []

        variants = []
        for index in time_indices[:2]:
            token = gloss_tokens[index]
            base = [item for i, item in enumerate(gloss_tokens) if i != index]
            if not base:
                continue

            front = [token] + base
            if front != gloss_tokens:
                variants.append((" ".join(front), zh_sentence))

            tail = base + [token]
            if tail != gloss_tokens:
                variants.append((" ".join(tail), zh_sentence))

            mid_insert = base[:1] + [token] + base[1:]
            if mid_insert != gloss_tokens:
                variants.append((" ".join(mid_insert), zh_sentence))

        return self._dedupe_variants(variants)

    def _negation_shift(self, gloss_tokens: List[str], zh_sentence: str) -> List[Tuple[str, str]]:
        """Perturb negation token positions to create correction pairs."""
        neg_indices = [index for index, token in enumerate(gloss_tokens) if token in self.neg_tokens]
        if not neg_indices:
            return []

        variants = []
        for index in neg_indices[:2]:
            neg_token = gloss_tokens[index]
            base = [item for i, item in enumerate(gloss_tokens) if i != index]
            if not base:
                continue

            tail = base + [neg_token]
            if tail != gloss_tokens:
                variants.append((" ".join(tail), zh_sentence))

            verb_index = self._find_first_verb_index(base)
            if verb_index >= 0:
                before_verb = base[:verb_index] + [neg_token] + base[verb_index:]
                if before_verb != gloss_tokens:
                    variants.append((" ".join(before_verb), zh_sentence))

            after_subject = base[:1] + [neg_token] + base[1:]
            if after_subject != gloss_tokens:
                variants.append((" ".join(after_subject), zh_sentence))

        return self._dedupe_variants(variants)

    def _backtrans_sim(self, gloss_tokens: List[str], zh_sentence: str) -> List[Tuple[str, str]]:
        """Simulate rule-based back-translation without external APIs."""
        zh_tokens = [token for token in self._tokenize_zh(zh_sentence) if token not in FUNCTION_WORDS]
        if not zh_tokens:
            return []

        buckets = {
            "subject": [],
            "time": [],
            "location": [],
            "neg": [],
            "verb": [],
            "object": [],
        }
        for token in zh_tokens:
            if token in self.time_tokens:
                buckets["time"].append(token)
            elif token in self.location_tokens:
                buckets["location"].append(token)
            elif token in self.neg_tokens:
                buckets["neg"].append(token)
            elif token in SUBJECT_CANDIDATES and not buckets["subject"]:
                buckets["subject"].append(token)
            elif self._looks_like_verb(token):
                buckets["verb"].append(token)
            else:
                buckets["object"].append(token)

        canonical = (
            buckets["subject"]
            + buckets["time"]
            + buckets["location"]
            + buckets["neg"]
            + buckets["verb"]
            + buckets["object"]
        )
        if not canonical:
            return []
        canonical_gloss = " ".join(canonical)
        if canonical == gloss_tokens:
            return []
        return [(canonical_gloss, zh_sentence)]

    def _subsequence_sampling(self, gloss_tokens: List[str], zh_sentence: str) -> List[Tuple[str, str]]:
        """Sample contiguous subsequences for long-sentence stabilization."""
        token_count = len(gloss_tokens)
        if token_count < 4:
            return []

        zh_tokens = self._tokenize_zh(zh_sentence)
        if not zh_tokens:
            return []

        sample_count = min(4, max(1, token_count // 3))
        variants = []
        for _ in range(sample_count):
            max_span = min(8, token_count)
            span = self.rng.randint(3, max_span)
            start = self.rng.randint(0, token_count - span)
            end = start + span
            gloss_piece = gloss_tokens[start:end]

            zh_start = int(round(start / token_count * len(zh_tokens)))
            zh_end = int(round(end / token_count * len(zh_tokens)))
            zh_end = max(zh_start + 1, min(len(zh_tokens), zh_end))
            zh_piece_tokens = zh_tokens[zh_start:zh_end]
            if not zh_piece_tokens:
                continue
            variants.append((" ".join(gloss_piece), "".join(zh_piece_tokens)))
        return self._dedupe_variants(variants)

    def _synonym_replace(self, gloss_tokens: List[str], zh_sentence: str) -> List[Tuple[str, str]]:
        """Replace gloss tokens with synonyms while keeping semantics."""
        variants = []
        for index, token in enumerate(gloss_tokens):
            replacement = self.synonym_dict.get(token)
            if not replacement:
                continue
            candidate = list(gloss_tokens)
            candidate[index] = replacement
            variants.append((" ".join(candidate), zh_sentence))

        if len(gloss_tokens) >= 4:
            candidate = list(gloss_tokens)
            swapped = False
            for index, token in enumerate(candidate):
                replacement = self.synonym_dict.get(token)
                if replacement and self.rng.random() < 0.5:
                    candidate[index] = replacement
                    swapped = True
            if swapped:
                variants.append((" ".join(candidate), zh_sentence))
        return self._dedupe_variants(variants)

    @staticmethod
    def _dedupe_variants(variants: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        unique = []
        seen = set()
        for gloss, zh in variants:
            key = (gloss.strip(), zh.strip())
            if key in seen or not key[0] or not key[1]:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    def _find_first_verb_index(self, tokens: Sequence[str]) -> int:
        for index, token in enumerate(tokens):
            if self._looks_like_verb(token):
                return index
        return -1

    def _looks_like_verb(self, token: str) -> bool:
        if token in self.verb_tokens:
            return True
        return token.endswith(("买", "去", "看", "办", "学", "做", "说", "吃", "喝"))

    def _tokenize_zh(self, sentence: str) -> List[str]:
        cleaned = self._normalize_space(sentence)
        if not cleaned:
            return []
        if " " in cleaned:
            return [token for token in cleaned.split(" ") if token]
        if jieba is not None:
            return [token for token in jieba.lcut(cleaned, cut_all=False) if token.strip()]
        return [char for char in cleaned if char.strip()]

    def _infer_order_pattern(self, gloss_tokens: Sequence[str]) -> str:
        if not gloss_tokens:
            return "empty"

        time_indices = [index for index, token in enumerate(gloss_tokens) if token in self.time_tokens]
        neg_indices = [index for index, token in enumerate(gloss_tokens) if token in self.neg_tokens]
        verb_index = self._find_first_verb_index(gloss_tokens)

        if time_indices and time_indices[0] == 0:
            return "time_front"
        if neg_indices and neg_indices[-1] == len(gloss_tokens) - 1:
            return "neg_tail"
        if verb_index > 0 and verb_index < len(gloss_tokens) - 1:
            return "svo_like"
        if verb_index == len(gloss_tokens) - 1:
            return "verb_tail"
        return "other"


if __name__ == "__main__":
    samples = [
        ("我 昨天 买 苹果", "我昨天买了苹果"),
        ("他 不 去 学校", "他不去学校"),
        ("残疾人 申请 政府 补偿", "残疾人向政府申请补偿"),
    ]

    augmentor = WordOrderAugmentor(
        gloss_vocab=None,
        zh_vocab=None,
        time_tokens=SIGN_LANGUAGE_ORDER_RULES["time_tokens"],
        neg_tokens=SIGN_LANGUAGE_ORDER_RULES["neg_tokens"],
        synonym_dict=DEFAULT_SYNONYM_DICT,
        strategies=[
            "component_swap",
            "temporal_shift",
            "negation_shift",
            "subsequence_sampling",
            "backtrans_sim",
            "synonym_replace",
        ],
        augment_ratio=3.0,
        seed=42,
    )
    augmented = augmentor.augment_dataset(samples)
    print(f"原始: {len(samples)} 条 -> 增强后: {len(augmented)} 条")
    for gloss, zh in augmented[:6]:
        print(f"[{gloss}] -> {zh}")
    print(f"增强统计: {augmentor.last_stats}")


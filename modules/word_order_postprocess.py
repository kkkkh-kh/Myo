# Module description: Rule-based Chinese word-order postprocessing for gloss translation outputs.

from __future__ import annotations

import re
from typing import Dict, List, Tuple

try:
    import jieba
except ImportError:  # pragma: no cover - fallback only when jieba is unavailable.
    jieba = None


DEFAULT_TIME_WORDS = [
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
    "最近",
    "去年",
    "今年",
    "明年",
    "上周",
    "下周",
]

DEFAULT_NEG_WORDS = ["不", "没", "没有", "别", "不要", "未", "勿"]

DEFAULT_MEASURE_WORDS = {
    "人": "个",
    "老师": "位",
    "医生": "位",
    "学生": "个",
    "狗": "只",
    "猫": "只",
    "鱼": "条",
    "牛": "头",
    "书": "本",
    "纸": "张",
    "桌子": "张",
    "椅子": "把",
    "苹果": "个",
    "橙子": "个",
    "香蕉": "根",
    "房子": "栋",
    "学校": "所",
    "医院": "家",
    "公司": "家",
    "材料": "份",
    "申请": "份",
}

NUMERAL_WORDS = {
    "零",
    "一",
    "二",
    "两",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
    "百",
    "千",
    "万",
}

DEFAULT_VERB_WORDS = {
    "买",
    "去",
    "看",
    "说",
    "申请",
    "提交",
    "学习",
    "工作",
    "办理",
    "帮助",
    "补偿",
    "补助",
    "拿",
    "给",
    "吃",
    "喝",
    "做",
    "找",
}

SUBJECT_CANDIDATES = {"我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "大家", "学生", "老师"}


class WordOrderPostProcessor:
    """Rule-based postprocessor for common Chinese word-order issues."""

    def __init__(
        self,
        time_words: List[str] = None,
        neg_words: List[str] = None,
        measure_word_dict: Dict[str, str] = None,
        enabled_rules: List[str] = None,
        confidence_threshold: float = 0.8,
    ) -> None:
        """Initialize postprocessing rules and dictionaries.

        Args:
            time_words: Time adverbial lexicon.
            neg_words: Negation lexicon.
            measure_word_dict: Noun-to-measure-word mapping.
            enabled_rules: Enabled rule names. ``None`` enables all rules.
            confidence_threshold: Minimum confidence for subject/object补全.
        """
        self.time_words = set(time_words or DEFAULT_TIME_WORDS)
        self.neg_words = set(neg_words or DEFAULT_NEG_WORDS)
        self.measure_word_dict = dict(DEFAULT_MEASURE_WORDS)
        self.measure_word_dict.update(measure_word_dict or {})
        self.enabled_rules = set(enabled_rules) if enabled_rules else {
            "temporal_reorder",
            "negation_reorder",
            "measure_word_insert",
            "subject_object_check",
            "dedup",
        }
        self.confidence_threshold = float(confidence_threshold)
        self.verb_words = set(DEFAULT_VERB_WORDS)
        lexicon = set(self.time_words) | set(self.neg_words) | set(self.measure_word_dict.keys()) | self.verb_words | SUBJECT_CANDIDATES
        self._fallback_lexicon = sorted((item for item in lexicon if len(item) > 1), key=len, reverse=True)

    def process(
        self,
        sentence: str,
        source_gloss: str = None,
        confidence: float = 1.0,
    ) -> Tuple[str, List[str]]:
        """Postprocess one sentence.

        Args:
            sentence: Raw model output sentence.
            source_gloss: Optional source gloss string.
            confidence: Optional model confidence score.

        Returns:
            Tuple of ``(processed_sentence, triggered_rules)``.
        """
        tokens = self._tokenize(sentence)
        gloss_tokens = source_gloss.split() if source_gloss else []
        triggered: List[str] = []

        if "temporal_reorder" in self.enabled_rules:
            tokens, changed = self._rule_temporal_reorder(tokens)
            if changed:
                triggered.append("temporal_reorder")

        if "negation_reorder" in self.enabled_rules:
            tokens, changed = self._rule_negation_reorder(tokens)
            if changed:
                triggered.append("negation_reorder")

        if "measure_word_insert" in self.enabled_rules:
            tokens, changed = self._rule_measure_word_insert(tokens)
            if changed:
                triggered.append("measure_word_insert")

        if "subject_object_check" in self.enabled_rules and confidence >= self.confidence_threshold:
            tokens, changed = self._rule_subject_object_check(tokens, gloss_tokens)
            if changed:
                triggered.append("subject_object_check")

        if "dedup" in self.enabled_rules:
            tokens, changed = self._rule_dedup(tokens)
            if changed:
                triggered.append("dedup")

        return self._join_tokens(tokens), triggered

    def _rule_temporal_reorder(self, tokens: List[str]) -> Tuple[List[str], bool]:
        """Move time words before the main verb when needed."""
        if len(tokens) < 3:
            return tokens, False
        verb_index = self._find_first_verb(tokens)
        if verb_index < 0:
            return tokens, False

        for index, token in enumerate(tokens):
            if token in self.time_words and index > verb_index:
                updated = [item for i, item in enumerate(tokens) if i != index]
                new_verb = self._find_first_verb(updated)
                insert_at = max(1, new_verb if new_verb >= 0 else 1)
                updated.insert(insert_at, token)
                return updated, True
        return tokens, False

    def _rule_negation_reorder(self, tokens: List[str]) -> Tuple[List[str], bool]:
        """Move negation words before the nearest main verb."""
        if len(tokens) < 2:
            return tokens, False
        verb_index = self._find_first_verb(tokens)
        if verb_index < 0:
            return tokens, False

        for index, token in enumerate(tokens):
            if token in self.neg_words and index > verb_index:
                updated = [item for i, item in enumerate(tokens) if i != index]
                new_verb = self._find_first_verb(updated)
                insert_at = max(0, new_verb)
                updated.insert(insert_at, token)
                return updated, True
        return tokens, False

    def _rule_measure_word_insert(self, tokens: List[str]) -> Tuple[List[str], bool]:
        """Insert default measure words between numerals and nouns."""
        if len(tokens) < 2:
            return tokens, False

        measure_values = set(self.measure_word_dict.values())
        updated: List[str] = []
        changed = False
        index = 0
        while index < len(tokens):
            token = tokens[index]
            updated.append(token)
            if not self._is_numeral(token):
                index += 1
                continue

            if index + 1 < len(tokens) and tokens[index + 1] in measure_values:
                index += 1
                continue

            if index + 1 < len(tokens):
                noun = tokens[index + 1]
                measure = self.measure_word_dict.get(noun)
                if measure:
                    updated.append(measure)
                    changed = True
            index += 1
        return updated, changed

    def _rule_subject_object_check(
        self,
        tokens: List[str],
        gloss_tokens: List[str],
    ) -> Tuple[List[str], bool]:
        """补全明显缺失的主语或宾语。"""
        if not gloss_tokens:
            return tokens, False

        updated = list(tokens)
        changed = False

        if updated and updated[0] not in SUBJECT_CANDIDATES and gloss_tokens[0] in SUBJECT_CANDIDATES:
            updated.insert(0, gloss_tokens[0])
            changed = True
        elif not updated and gloss_tokens[0] in SUBJECT_CANDIDATES:
            updated.append(gloss_tokens[0])
            changed = True

        # Conservative object补全: only when tokenized at word level and object is clearly absent.
        if updated and any(len(token) > 1 for token in updated):
            gloss_verb_index = self._find_first_verb(gloss_tokens)
            if gloss_verb_index >= 0 and gloss_verb_index < len(gloss_tokens) - 1:
                candidate_objects = [token for token in gloss_tokens[gloss_verb_index + 1 :] if token not in self.neg_words]
                if candidate_objects:
                    likely_object = candidate_objects[0]
                    joined = "".join(updated)
                    if likely_object not in joined and likely_object not in updated:
                        updated.append(likely_object)
                        changed = True
        return updated, changed

    def _rule_dedup(self, tokens: List[str]) -> Tuple[List[str], bool]:
        """Remove adjacent or near-duplicate tokens."""
        if len(tokens) < 2:
            return tokens, False
        updated: List[str] = []
        changed = False
        for token in tokens:
            if updated and token == updated[-1]:
                changed = True
                continue
            if len(updated) >= 2 and token == updated[-2]:
                changed = True
                continue
            updated.append(token)
        return updated, changed

    def batch_process(
        self,
        sentences: List[str],
        source_glosses: List[str] = None,
    ) -> List[Tuple[str, List[str]]]:
        """Postprocess a batch of sentences."""
        source_glosses = source_glosses or [""] * len(sentences)
        results = []
        for sentence, gloss in zip(sentences, source_glosses):
            results.append(self.process(sentence, source_gloss=gloss))
        return results

    def _tokenize(self, sentence: str) -> List[str]:
        text = re.sub(r"\s+", " ", (sentence or "").strip())
        if not text:
            return []
        if " " in text:
            return [token for token in text.split(" ") if token]
        if jieba is not None:
            return [token for token in jieba.lcut(text, cut_all=False) if token.strip()]
        return self._greedy_tokenize_without_jieba(text)

    def _greedy_tokenize_without_jieba(self, text: str) -> List[str]:
        tokens: List[str] = []
        index = 0
        while index < len(text):
            matched = None
            for word in self._fallback_lexicon:
                if text.startswith(word, index):
                    matched = word
                    break
            if matched is None:
                tokens.append(text[index])
                index += 1
            else:
                tokens.append(matched)
                index += len(matched)
        return tokens

    @staticmethod
    def _join_tokens(tokens: List[str]) -> str:
        return "".join(token for token in tokens if token and token.strip())

    def _find_first_verb(self, tokens: List[str]) -> int:
        for index, token in enumerate(tokens):
            if self._is_verb(token):
                return index
        return -1

    def _is_verb(self, token: str) -> bool:
        if token in self.verb_words:
            return True
        return token.endswith(("买", "去", "看", "办", "做", "说", "吃", "喝", "学"))

    @staticmethod
    def _is_numeral(token: str) -> bool:
        if token in NUMERAL_WORDS:
            return True
        return bool(re.fullmatch(r"\d+", token))


if __name__ == "__main__":
    processor = WordOrderPostProcessor()
    tests = [
        ("我买苹果昨天", "我 昨天 买 苹果"),
        ("我买苹果没有", "我 没有 买 苹果"),
        ("三苹果", "三 苹果"),
        ("我我买苹果", "我 买 苹果"),
    ]
    for sentence, gloss in tests:
        result, rules = processor.process(sentence, gloss)
        print(f"输入: {sentence}")
        print(f"输出: {result}  触发规则: {rules}\n")

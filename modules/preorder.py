import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReorderRule:
    pattern: List[str]
    replacement: List[str]
    example_gloss: str
    example_chinese: str
    priority: int


class PreorderModule:
    """Rule-based pre-ordering for gloss tokens."""

    CATEGORY_LEXICONS: Dict[str, set] = {
        "TIME": {"昨天", "今天", "明天", "刚才", "刚刚", "以后", "以前", "现在", "马上", "上周", "下周", "去年", "明年"},
        "VERB": {"买", "去", "来", "申请", "提交", "交", "补偿", "看", "做", "办理", "完成", "参考", "设计", "布局", "开会", "学习", "帮助", "查询"},
        "SUBJ": {"我", "你", "他", "她", "我们", "你们", "他们", "残疾人", "学生", "老师", "工作人员"},
        "OBJ": {"苹果", "材料", "补偿", "手续", "申请", "设计", "布局", "文件", "政府", "医院", "学校", "方案"},
        "NEG": {"不", "没", "没有", "别"},
        "ASPECT": {"了", "过", "着", "完", "好"},
        "QWORD": {"谁", "什么", "哪里", "哪儿", "怎么", "为什么", "几", "多少"},
        "LOC": {"北京", "上海", "政府", "医院", "学校", "家", "办公室", "窗口"},
        "MODAL": {"能", "可以", "会", "应该", "要"},
        "ADJ": {"红", "大", "小", "新", "旧", "漂亮", "方便", "重要", "残疾", "详细"},
        "NOUN": {"苹果", "衣服", "材料", "窗口", "老师", "学生", "方案", "房子", "桌子", "问题", "申请表", "补偿金"},
        "ADV": {"已经", "正在", "马上", "先", "再", "都", "也", "一起"},
        "PRON": {"我", "你", "他", "她", "它", "我们", "你们", "他们"},
        "AUX": {"把", "被", "给", "向", "对"},
        "NUM": {"一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "两"},
    }

    def __init__(self, rules_path: str = None) -> None:
        default_path = Path(__file__).resolve().parent.parent / "data" / "reorder_rules.json"
        self.rules_path = Path(rules_path) if rules_path else default_path
        self.rules = self._load_rules()

    def _load_rules(self) -> List[ReorderRule]:
        with self.rules_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        rules = [ReorderRule(**item) for item in payload]
        return sorted(rules, key=lambda rule: rule.priority, reverse=True)

    def _categorize(self, token: str) -> str:
        if not token:
            return "UNK"
        for category, lexicon in self.CATEGORY_LEXICONS.items():
            if token in lexicon:
                return category
        if token.isdigit() or token in self.CATEGORY_LEXICONS["NUM"]:
            return "NUM"
        if token.endswith("吗") or token.endswith("呢"):
            return "QWORD"
        if token in {"，", "。", "？", "！", ",", ".", "?", "!"}:
            return "PUNCT"
        return "UNK"

    def _match_pattern(self, window: Sequence[str], pattern: Sequence[str]) -> bool:
        if len(window) != len(pattern):
            return False
        for token, expected in zip(window, pattern):
            category = self._categorize(token)
            if expected != category and expected != token:
                return False
        return True

    def _rebuild_window(self, window: Sequence[str], pattern: Sequence[str], replacement: Sequence[str]) -> List[str]:
        bucket: Dict[str, List[str]] = {}
        for token, category in zip(window, pattern):
            bucket.setdefault(category, []).append(token)
        rebuilt: List[str] = []
        used_counts: Dict[str, int] = {}
        for item in replacement:
            values = bucket.get(item)
            if values:
                index = used_counts.get(item, 0)
                rebuilt.append(values[min(index, len(values) - 1)])
                used_counts[item] = index + 1
            else:
                rebuilt.append(item)
        return rebuilt

    def reorder(self, gloss_tokens: List[str]) -> List[str]:
        if not gloss_tokens:
            return []
        if len(gloss_tokens) == 1:
            return list(gloss_tokens)

        reordered = list(gloss_tokens)
        max_passes = 3
        for _ in range(max_passes):
            changed = False
            for rule in self.rules:
                pattern_len = len(rule.pattern)
                if pattern_len == 0 or pattern_len > len(reordered):
                    continue
                start = 0
                while start <= len(reordered) - pattern_len:
                    window = reordered[start : start + pattern_len]
                    if self._match_pattern(window, rule.pattern):
                        new_window = self._rebuild_window(window, rule.pattern, rule.replacement)
                        if new_window != list(window):
                            LOGGER.debug(
                                "预排序命中规则 priority=%s, before=%s, after=%s",
                                rule.priority,
                                " ".join(window),
                                " ".join(new_window),
                            )
                            reordered[start : start + pattern_len] = new_window
                            changed = True
                            start += pattern_len
                            continue
                    start += 1
            if not changed:
                break
        return reordered

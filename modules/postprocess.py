import re
from typing import List


class PostProcessor:
    """Post-process decoder tokens into a readable Chinese sentence."""

    PUNCTUATION_MAP = {
        ",": "，",
        ".": "。",
        "!": "！",
        "?": "？",
        ";": "；",
        ":": "：",
    }

    def _normalize_punctuation(self, text: str) -> str:
        for source, target in self.PUNCTUATION_MAP.items():
            text = text.replace(source, target)
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"([，。！？；：]){2,}", lambda match: match.group(1)[0], text)
        text = re.sub(r"([，。！？；：])\s+", r"\1", text)
        text = re.sub(r"\s+([，。！？；：])", r"\1", text)
        return text.strip("，。！？；：")

    def _looks_character_level(self, tokens: List[str]) -> bool:
        signal_tokens = [token for token in tokens if token not in self.PUNCTUATION_MAP]
        if not signal_tokens:
            return True
        single_char_tokens = sum(1 for token in signal_tokens if len(token) == 1)
        return single_char_tokens / max(1, len(signal_tokens)) >= 0.8

    def process(self, tokens: List[str]) -> str:
        cleaned_tokens = [token.strip() for token in tokens if token and token.strip()]
        if not cleaned_tokens:
            return ""
        if self._looks_character_level(cleaned_tokens):
            return self._normalize_punctuation("".join(cleaned_tokens))
        return self._normalize_punctuation("".join(cleaned_tokens))

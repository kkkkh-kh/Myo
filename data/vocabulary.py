import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union


TextLike = Union[str, Sequence[str]]


class Vocabulary:
    """Vocabulary with JSON serialization and character fallback."""

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, token_to_id: Dict[str, int] = None) -> None:
        base_mapping = {
            self.PAD_TOKEN: self.PAD_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }
        if token_to_id is not None:
            base_mapping.update(token_to_id)
        self.token_to_id: Dict[str, int] = dict(sorted(base_mapping.items(), key=lambda item: item[1]))
        self.id_to_token: List[str] = [None] * len(self.token_to_id)
        for token, index in self.token_to_id.items():
            if index >= len(self.id_to_token):
                self.id_to_token.extend([self.UNK_TOKEN] * (index - len(self.id_to_token) + 1))
            self.id_to_token[index] = token

    @staticmethod
    def _normalize_tokens(text: TextLike) -> List[str]:
        if isinstance(text, str):
            stripped = text.strip()
            if not stripped:
                return []
            if " " in stripped:
                return [token for token in stripped.split() if token]
            return [char for char in stripped if char.strip()]
        return [str(token).strip() for token in text if str(token).strip()]

    def build_from_corpus(self, texts: Iterable[TextLike], max_size: int) -> None:
        """Build a frequency-pruned vocabulary from tokenized texts."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._normalize_tokens(text))

        reserved = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        capacity = max(0, max_size - len(reserved))
        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        self.token_to_id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }
        for token, _ in sorted_tokens[:capacity]:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token = [None] * len(self.token_to_id)
        for token, index in self.token_to_id.items():
            self.id_to_token[index] = token

    def save(self, path: Union[str, Path]) -> None:
        payload = {
            "token_to_id": self.token_to_id,
            "special_tokens": {
                "pad": self.PAD_TOKEN,
                "bos": self.BOS_TOKEN,
                "eos": self.EOS_TOKEN,
                "unk": self.UNK_TOKEN,
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        token_to_id = {str(token): int(index) for token, index in payload["token_to_id"].items()}
        return cls(token_to_id=token_to_id)

    def add_token(self, token: str) -> int:
        token = token.strip()
        if not token:
            return self.PAD_ID
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def encode(self, text: TextLike, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = self._normalize_tokens(text)
        ids: List[int] = []
        if add_bos:
            ids.append(self.BOS_ID)
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
                continue
            if len(token) > 1:
                sub_ids = [self.token_to_id.get(char, self.UNK_ID) for char in token]
                ids.extend(sub_ids)
            else:
                ids.append(self.UNK_ID)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens: List[str] = []
        for index in ids:
            if index < 0 or index >= len(self.id_to_token):
                token = self.UNK_TOKEN
            else:
                token = self.id_to_token[index]
            if skip_special_tokens and token in {
                self.PAD_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
                self.UNK_TOKEN,
            }:
                if token == self.UNK_TOKEN:
                    tokens.append(token)
                continue
            tokens.append(token)
        return " ".join(tokens).strip()

    def to_tokens(self, ids: Sequence[int], skip_special_tokens: bool = True) -> List[str]:
        decoded = self.decode(ids, skip_special_tokens=skip_special_tokens)
        return decoded.split() if decoded else []

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"

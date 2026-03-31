import csv
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import jieba


FULLWIDTH_PUNCTUATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "？": "?",
        "！": "!",
        "：": ":",
        "；": ";",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)

ASCII_TO_FULLWIDTH_PUNCTUATION = {
    ",": "，",
    ".": "。",
    "?": "？",
    "!": "！",
    ":": "：",
    ";": "；",
}

SPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_TOKENS = {",", ".", ";", ":", "!", "?", "(", ")", "[", "]"}
LEADING_DIGITS_PATTERN = re.compile(r"^(\d+)(.+)$")


def normalize_punctuation(text: str) -> str:
    return text.translate(FULLWIDTH_PUNCTUATION)


def clean_gloss_text(text: str) -> str:
    """Normalize gloss text while keeping token boundaries explicit."""
    text = normalize_punctuation(text.strip())
    text = text.replace("/", " ").replace("|", " ")
    text = re.sub(r"[^\w\s\u4e00-\u9fff?!,.;:\-]", " ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def clean_chinese_text(text: str) -> str:
    """Normalize Chinese text for training and inference."""
    text = normalize_punctuation(text.strip())
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"([,.;:!?()\[\]])", r" \1 ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def merge_number_tokens(tokens: Sequence[str]) -> List[str]:
    """Merge adjacent numeric gloss tokens into one token.

    Examples:
      ['2', '0', '2', '3'] -> ['2023']
      ['2', '0', '2', '3年'] -> ['2023', '年']
      ['1', '0', '0', '0', '万'] -> ['1000', '万']
    """
    merged: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            merged.append("".join(buffer))
            buffer = []

    for token in tokens:
        if not token:
            continue

        if re.fullmatch(r"\d+", token):
            buffer.append(token)
            continue

        # Handle attached suffix such as "3年":
        # when a numeric buffer exists, merge the leading digits first.
        leading_match = LEADING_DIGITS_PATTERN.match(token)
        if leading_match is not None:
            leading_digits, suffix = leading_match.groups()
            if buffer:
                buffer.append(leading_digits)
                flush_buffer()
                if suffix.strip():
                    merged.append(suffix)
                continue
            # Without an active buffer, keep the original token unchanged.

        flush_buffer()
        merged.append(token)

    flush_buffer()
    return merged


def tokenize_gloss(text: str) -> List[str]:
    cleaned = clean_gloss_text(text)
    tokens = cleaned.split() if cleaned else []
    return merge_number_tokens(tokens)


def tokenize_chinese(text: str, mode: str = "char") -> List[str]:
    cleaned = clean_chinese_text(text)
    if not cleaned:
        return []
    if mode == "char":
        chars = [char for char in cleaned if not char.isspace()]
        return merge_number_tokens(chars) 
    if mode != "jieba":
        raise ValueError(f"Unsupported Chinese tokenization mode: {mode}")

    pieces: List[str] = []
    for piece in cleaned.split():
        if piece in PUNCTUATION_TOKENS:
            pieces.append(piece)
        else:
            pieces.extend(token for token in jieba.lcut(piece, cut_all=False) if token.strip())
    return merge_number_tokens(pieces)


def detokenize_chinese(tokens: Sequence[str]) -> str:
    text = "".join(token for token in tokens if token and str(token).strip())
    text = re.sub(r"\s+", "", text)
    for ascii_punctuation, fullwidth_punctuation in ASCII_TO_FULLWIDTH_PUNCTUATION.items():
        text = text.replace(ascii_punctuation, fullwidth_punctuation)
    return text


def _normalize_header(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


def _resolve_csv_columns(fieldnames: Sequence[str]) -> Tuple[str, str]:
    normalized_to_original = {
        _normalize_header(fieldname): fieldname
        for fieldname in fieldnames
        if fieldname is not None and fieldname.strip()
    }
    gloss_candidates = ["gloss", "glosssequence", "glosssentence", "source"]
    chinese_candidates = [
        "chinesesentences",
        "chinesesentence",
        "chinese",
        "sentence",
        "target",
        "translation",
    ]

    gloss_column = next(
        (normalized_to_original[candidate] for candidate in gloss_candidates if candidate in normalized_to_original),
        None,
    )
    chinese_column = next(
        (normalized_to_original[candidate] for candidate in chinese_candidates if candidate in normalized_to_original),
        None,
    )
    if gloss_column is None or chinese_column is None:
        raise ValueError(f"Unsupported CSV columns: {list(fieldnames)}")
    return gloss_column, chinese_column


def _read_csv_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        gloss_column, chinese_column = _resolve_csv_columns(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            gloss = (row.get(gloss_column) or "").strip()
            chinese = (row.get(chinese_column) or "").strip()
            if not gloss and not chinese:
                continue
            if not gloss or not chinese:
                raise ValueError(f"Incomplete CSV row on line {row_number}: {row}")
            pairs.append((clean_gloss_text(gloss), clean_chinese_text(chinese)))
    return pairs


def _read_tsv_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid TSV format on line {line_number}: {raw_line!r}")
            gloss, chinese = parts
            if line_number == 1:
                normalized_first = _normalize_header(gloss)
                normalized_second = _normalize_header(chinese)
                if "gloss" in normalized_first and "chinese" in normalized_second:
                    continue
            pairs.append((clean_gloss_text(gloss), clean_chinese_text(chinese)))
    return pairs


def read_parallel_pairs(path: str) -> List[Tuple[str, str]]:
    """Read gloss-Chinese pairs from either TSV or CSV files."""
    data_path = Path(path)
    if data_path.suffix.lower() == ".csv":
        return _read_csv_pairs(data_path)
    return _read_tsv_pairs(data_path)


def read_tsv_pairs(path: str) -> List[Tuple[str, str]]:
    """Backward-compatible alias that now supports both TSV and CSV files."""
    return read_parallel_pairs(path)


# 在 extract_corpora 里临时加一行 debug
def extract_corpora(pairs, zh_tokenizer_mode="char"):
    gloss_texts, chinese_texts = [], []
    for i, (gloss, chinese) in enumerate(pairs):
        g_tokens = tokenize_gloss(gloss)
        c_tokens = tokenize_chinese(chinese, mode=zh_tokenizer_mode)
        if i < 3:
            print(f"[DEBUG] gloss raw: {gloss!r}")
            print(f"[DEBUG] gloss tokens: {g_tokens}")
            print(f"[DEBUG] chinese raw: {chinese!r}")
            print(f"[DEBUG] chinese tokens: {c_tokens}")
            print("---")
        gloss_texts.append(g_tokens)
        chinese_texts.append(c_tokens)
    return gloss_texts, chinese_texts

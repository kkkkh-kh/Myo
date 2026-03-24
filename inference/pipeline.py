from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import yaml

from data.preprocess import tokenize_gloss
from data.vocabulary import Vocabulary
from modules.postprocess import PostProcessor
from modules.preorder import PreorderModule


class TranslationPipeline:
    """End-to-end ONNX inference pipeline."""

    def __init__(self, model_dir: str, config_path: str) -> None:
        self.model_dir = Path(model_dir)
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.max_seq_len = int(self.config["deploy"].get("max_seq_len", 32))
        self.max_decode_len = int(self.config.get("data", {}).get("max_zh_len", 48))
        self.memory_limit_mb = float(self.config["deploy"].get("memory_limit_mb", 60))
        self.num_layers = int(self.config["model"].get("num_layers", 2))
        self.hidden_dim = int(self.config["model"].get("hidden_dim", 256))
        self.preorder = PreorderModule()
        self.postprocess = PostProcessor()
        self.gloss_vocab: Optional[Vocabulary] = None
        self.zh_vocab: Optional[Vocabulary] = None
        self.encoder_session: Optional[ort.InferenceSession] = None
        self.decoder_session: Optional[ort.InferenceSession] = None

    def _session_options(self) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False
        return options

    def _resolve_model_path(self, stem: str) -> Path:
        candidates = [self.model_dir / f"{stem}.int8.onnx", self.model_dir / f"{stem}.onnx"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"未找到模型文件：{candidates[0]} 或 {candidates[1]}")

    def _ensure_loaded(self) -> None:
        if self.encoder_session is not None and self.decoder_session is not None:
            return
        gloss_vocab_path = self.model_dir / "gloss_vocab.json"
        zh_vocab_path = self.model_dir / "zh_vocab.json"
        if not gloss_vocab_path.exists() or not zh_vocab_path.exists():
            raise FileNotFoundError("未找到词表文件，请先导出 gloss_vocab.json 和 zh_vocab.json。")

        self.gloss_vocab = Vocabulary.load(gloss_vocab_path)
        self.zh_vocab = Vocabulary.load(zh_vocab_path)
        options = self._session_options()
        self.encoder_session = ort.InferenceSession(
            self._resolve_model_path("encoder").as_posix(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.decoder_session = ort.InferenceSession(
            self._resolve_model_path("decoder").as_posix(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _prepare_gloss_ids(self, gloss_sequence: str) -> np.ndarray:
        tokens = self.preorder.reorder(tokenize_gloss(gloss_sequence))
        token_ids = self.gloss_vocab.encode(tokens, add_eos=True)[: self.max_seq_len]
        if not token_ids:
            token_ids = [Vocabulary.EOS_ID]
        return np.asarray([token_ids], dtype=np.int64)

    def _decode_ids(self, predicted_ids: List[int]) -> List[str]:
        tokens = []
        for index in predicted_ids:
            if index == Vocabulary.EOS_ID:
                break
            if index in {Vocabulary.PAD_ID, Vocabulary.BOS_ID}:
                continue
            if index < len(self.zh_vocab.id_to_token):
                tokens.append(self.zh_vocab.id_to_token[index])
            else:
                tokens.append(Vocabulary.UNK_TOKEN)
        return tokens

    def translate(self, gloss_sequence: str) -> str:
        self._ensure_loaded()
        input_ids = self._prepare_gloss_ids(gloss_sequence)
        src_mask = input_ids != Vocabulary.PAD_ID
        enc_output, enc_hidden = self.encoder_session.run(None, {"input_ids": input_ids})
        hidden = np.repeat(enc_hidden[np.newaxis, :, :], self.num_layers, axis=0).astype(np.float32)
        input_token = np.asarray([Vocabulary.BOS_ID], dtype=np.int64)
        predicted_ids: List[int] = []

        for _ in range(self.max_decode_len):
            logits, hidden, _ = self.decoder_session.run(
                None,
                {
                    "input_token": input_token,
                    "hidden": hidden,
                    "enc_output": enc_output,
                    "src_mask": src_mask,
                },
            )
            next_token = int(np.argmax(logits, axis=-1)[0])
            if next_token == Vocabulary.EOS_ID:
                break
            predicted_ids.append(next_token)
            input_token = np.asarray([next_token], dtype=np.int64)

        return self.postprocess.process(self._decode_ids(predicted_ids))

    def batch_translate(self, gloss_list: List[str]) -> List[str]:
        self._ensure_loaded()
        return [self.translate(gloss) for gloss in gloss_list]

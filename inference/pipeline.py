import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import yaml

from data.preprocess import tokenize_gloss,merge_number_tokens
from data.vocabulary import Vocabulary
from modules.postprocess import PostProcessor
from modules.preorder import PreorderModule
from modules.word_order_postprocess import DEFAULT_NEG_WORDS, DEFAULT_TIME_WORDS, WordOrderPostProcessor


LOGGER = logging.getLogger(__name__)


class TranslationPipeline:
    """End-to-end ONNX inference pipeline."""

    def __init__(self, model_dir: str, config_path: str, enable_postprocess: Optional[bool] = None) -> None:
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

        postprocess_cfg = self.config.get("postprocess", {})
        self.enable_word_order_postprocess = (
            bool(postprocess_cfg.get("enabled", False)) if enable_postprocess is None else bool(enable_postprocess)
        )
        word_order_cfg = self.config.get("word_order_augment", {})
        self.word_order_postprocess = WordOrderPostProcessor(
            time_words=word_order_cfg.get("time_tokens", DEFAULT_TIME_WORDS),
            neg_words=word_order_cfg.get("neg_tokens", DEFAULT_NEG_WORDS),
            enabled_rules=postprocess_cfg.get("enabled_rules", None),
            confidence_threshold=float(postprocess_cfg.get("confidence_threshold", 0.8)),
        )

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
        raise FileNotFoundError(f"Model file not found: {candidates[0]} or {candidates[1]}")

    def _ensure_loaded(self) -> None:
        if self.encoder_session is not None and self.decoder_session is not None:
            return
        gloss_vocab_path = self.model_dir / "gloss_vocab.json"
        zh_vocab_path = self.model_dir / "zh_vocab.json"
        if not gloss_vocab_path.exists() or not zh_vocab_path.exists():
            raise FileNotFoundError("Vocabulary files not found. Export gloss_vocab.json and zh_vocab.json first.")

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
        tokens = tokenize_gloss(gloss_sequence)
        tokens = merge_number_tokens(gloss_sequence)
        tokens = self.preorder.reorder(tokens)
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

    def _apply_generation_constraints(
            self, 
            logits: np.ndarray, 
            step: int,
            predicted_ids:Optional[List[int]]=None,
            repetition_penalty:float=1.3,
            no_repeat_ngram_size:int =3
            ) -> np.ndarray:
        constrained = np.array(logits, copy=True)
        constrained[..., Vocabulary.PAD_ID] = -np.inf
        constrained[..., Vocabulary.BOS_ID] = -np.inf
        if step < 1:
            constrained[..., Vocabulary.EOS_ID] = -np.inf
        if predicted_ids:
            if repetition_penalty != 1.0:
                for token_id in set(predicted_ids):
                    if constrained[..., token_id] > 0:
                        constrained[..., token_id] /= repetition_penalty
                    else:
                        constrained[..., token_id] *= repetition_penalty
            if no_repeat_ngram_size > 1 and len(predicted_ids) >= no_repeat_ngram_size - 1:
                ngram_prefix = tuple(predicted_ids[-(no_repeat_ngram_size - 1):])
                for i in range(len(predicted_ids) - no_repeat_ngram_size + 1):
                    if tuple(predicted_ids[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                        banned_token = predicted_ids[i + no_repeat_ngram_size - 1]
                        constrained[..., banned_token] = -np.inf
        return constrained

    def translate(self, gloss_sequence: str) -> str:
        self._ensure_loaded()
        input_ids = self._prepare_gloss_ids(gloss_sequence)
        src_mask = input_ids != Vocabulary.PAD_ID
        enc_output, enc_hidden = self.encoder_session.run(None, {"input_ids": input_ids})
        hidden = np.repeat(enc_hidden[np.newaxis, :, :], self.num_layers, axis=0).astype(np.float32)
        input_token = np.asarray([Vocabulary.BOS_ID], dtype=np.int64)
        predicted_ids: List[int] = []

        for step in range(self.max_decode_len):
            logits, hidden, _ = self.decoder_session.run(
                None,
                {
                    "input_token": input_token,
                    "hidden": hidden,
                    "enc_output": enc_output,
                    "src_mask": src_mask,
                },
            )
            constrained_logits = self._apply_generation_constraints(
            logits,
            step,
            predicted_ids=predicted_ids,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
            next_token = int(np.argmax(constrained_logits, axis=-1)[0])
            if next_token == Vocabulary.EOS_ID:
                break
            predicted_ids.append(next_token)
            input_token = np.asarray([next_token], dtype=np.int64)

        raw_sentence = self.postprocess.process(self._decode_ids(predicted_ids))
        if not self.enable_word_order_postprocess:
            return raw_sentence

        final_sentence, triggered_rules = self.word_order_postprocess.process(raw_sentence, source_gloss=gloss_sequence)
        if triggered_rules:
            LOGGER.debug("后处理触发规则: %s", triggered_rules)
        return final_sentence

    def batch_translate(self, gloss_list: List[str]) -> List[str]:
        self._ensure_loaded()
        return [self.translate(gloss) for gloss in gloss_list]

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

ROOT_DIR = Path(os.environ.get("ROOT_DIR", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "datasets"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", ROOT_DIR / "configs" / "plm_mt5_small.yaml"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT_DIR / "checkpoints" / "plm_baseline"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from data.preprocess import clean_chinese_text
from train.evaluate import compute_bleu4, compute_rouge_l, compute_wer
from train.plm_utils import (
    ApproxSemanticAwareLabelSmoother,
    ParallelExample,
    build_semantic_neighbor_map,
    collect_source_tokens,
    format_gloss_prompt,
    load_training_examples,
    read_parallel_examples,
)


def _require_transformers():
    try:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "train_plm.py requires transformers, accelerate, and sentencepiece. "
            "Install them from requirements/environment.yml first."
        ) from exc
    return AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed


def resolve_dataset_path(data_dir: Path, *candidates: str) -> Path:
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No dataset file found. Tried: {', '.join(str(data_dir / candidate) for candidate in candidates)}"
    )


class TokenizedGlossDataset(Dataset):
    def __init__(
        self,
        examples: List[ParallelExample],
        tokenizer,
        *,
        max_source_length: int,
        max_target_length: int,
        task_prefix: str,
        include_source_tags: bool,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.task_prefix = task_prefix
        self.include_source_tags = include_source_tags

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        example = self.examples[index]
        model_input = format_gloss_prompt(
            example.gloss,
            task_prefix=self.task_prefix,
            source=example.source,
            include_source_tag=self.include_source_tags,
        )
        source = self.tokenizer(
            model_input,
            truncation=True,
            max_length=self.max_source_length,
        )
        target = self.tokenizer(
            text_target=example.text,
            truncation=True,
            max_length=self.max_target_length,
        )
        source["labels"] = target["input_ids"]
        return source


@dataclass
class RuntimeArtifacts:
    trainer: Any
    tokenizer: Any
    model: Any
    train_examples: List[ParallelExample]
    eval_examples: List[ParallelExample]


def _build_runtime(config: Dict[str, Any]) -> RuntimeArtifacts:
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed = _require_transformers()

    model_config = config.get("model", {})
    data_config = config.get("data", {})
    train_config = config.get("train", {})
    semantic_config = train_config.get("semantic_label_smoothing", {})

    set_seed(int(train_config.get("seed", 42)))

    train_path = Path(data_config.get("train_path") or resolve_dataset_path(DATA_DIR, "train.tsv", "train.csv"))
    val_path = Path(data_config.get("val_path") or resolve_dataset_path(DATA_DIR, "val.tsv", "dev.tsv", "val.csv", "dev.csv"))
    synthetic_paths = [str(path) for path in data_config.get("synthetic_paths", [])]

    train_examples = load_training_examples(
        train_path.as_posix(),
        synthetic_paths=synthetic_paths,
        max_synthetic_ratio=float(data_config.get("max_synthetic_ratio", 1.0)),
        seed=int(train_config.get("seed", 42)),
    )
    eval_examples = read_parallel_examples(val_path.as_posix(), default_source="validation")

    tokenizer = AutoTokenizer.from_pretrained(model_config["name_or_path"], use_fast=model_config.get("use_fast", True))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_config["name_or_path"])

    include_source_tags = bool(data_config.get("include_source_tags", False))
    if include_source_tags:
        special_tokens = collect_source_tokens(train_examples + eval_examples)
        if special_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            model.resize_token_embeddings(len(tokenizer))

    train_dataset = TokenizedGlossDataset(
        train_examples,
        tokenizer,
        max_source_length=int(model_config.get("max_source_length", 96)),
        max_target_length=int(model_config.get("max_target_length", 96)),
        task_prefix=str(data_config.get("task_prefix", "translate gloss to chinese:")),
        include_source_tags=include_source_tags,
    )
    eval_dataset = TokenizedGlossDataset(
        eval_examples,
        tokenizer,
        max_source_length=int(model_config.get("max_source_length", 96)),
        max_target_length=int(model_config.get("max_target_length", 96)),
        task_prefix=str(data_config.get("task_prefix", "translate gloss to chinese:")),
        include_source_tags=include_source_tags,
    )

    semantic_smoother = None
    if semantic_config.get("enabled", False):
        neighbor_map = build_semantic_neighbor_map(
            [example.text for example in train_examples],
            encode_surface_token=lambda token: tokenizer(token, add_special_tokens=False)["input_ids"],
            zh_tokenizer_mode=str(semantic_config.get("tokenizer_mode", "jieba")),
            top_k=int(semantic_config.get("top_k", 4)),
            min_similarity=float(semantic_config.get("min_similarity", 0.5)),
        )
        semantic_smoother = ApproxSemanticAwareLabelSmoother(
            neighbor_map,
            smoothing=float(semantic_config.get("smoothing", 0.1)),
            ignore_index=-100,
        )

    def compute_metrics(eval_prediction) -> Dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = [clean_chinese_text(text) for text in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
        decoded_labels = [clean_chinese_text(text) for text in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return {
            "bleu4": compute_bleu4(decoded_predictions, decoded_labels),
            "rouge_l": compute_rouge_l(decoded_predictions, decoded_labels),
            "wer": compute_wer(decoded_predictions, decoded_labels),
        }

    class GlossSeq2SeqTrainer(Seq2SeqTrainer):
        def __init__(self, *args, semantic_label_smoother=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.semantic_label_smoother = semantic_label_smoother

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            if labels is None or self.semantic_label_smoother is None:
                loss = outputs.loss
            else:
                loss = self.semantic_label_smoother(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    output_dir = Path(train_config.get("output_dir", OUTPUT_DIR.as_posix()))
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir.as_posix(),
        predict_with_generate=True,
        evaluation_strategy=str(train_config.get("evaluation_strategy", "epoch")),
        save_strategy=str(train_config.get("save_strategy", "epoch")),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(train_config.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_config.get("learning_rate", 3.0e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.1)),
        num_train_epochs=float(train_config.get("num_train_epochs", 10)),
        logging_steps=int(train_config.get("logging_steps", 20)),
        save_total_limit=int(train_config.get("save_total_limit", 2)),
        load_best_model_at_end=bool(train_config.get("load_best_model_at_end", True)),
        metric_for_best_model=str(train_config.get("metric_for_best_model", "bleu4")),
        greater_is_better=bool(train_config.get("greater_is_better", True)),
        generation_num_beams=int(train_config.get("generation_num_beams", 4)),
        generation_max_length=int(model_config.get("max_target_length", 96)),
        fp16=bool(train_config.get("fp16", False)),
        report_to=list(train_config.get("report_to", [])),
        label_smoothing_factor=0.0 if semantic_smoother is not None else float(train_config.get("label_smoothing_factor", 0.0)),
    )

    trainer = GlossSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
        semantic_label_smoother=semantic_smoother,
    )
    return RuntimeArtifacts(
        trainer=trainer,
        tokenizer=tokenizer,
        model=model,
        train_examples=train_examples,
        eval_examples=eval_examples,
    )


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config.setdefault("train", {})
    config["train"].setdefault("output_dir", OUTPUT_DIR.as_posix())
    config.setdefault("run_id", os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S")))

    artifacts = _build_runtime(config)
    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "runtime_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "train_examples": len(artifacts.train_examples),
                "eval_examples": len(artifacts.eval_examples),
                "sources": sorted({example.source for example in artifacts.train_examples}),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"CONFIG_PATH : {CONFIG_PATH}")
    print(f"OUTPUT_DIR  : {output_dir}")
    print(f"Train size  : {len(artifacts.train_examples)}")
    print(f"Eval size   : {len(artifacts.eval_examples)}")

    train_result = artifacts.trainer.train()
    artifacts.trainer.save_model(output_dir.as_posix())
    artifacts.tokenizer.save_pretrained(output_dir.as_posix())

    metrics = artifacts.trainer.evaluate(metric_key_prefix="eval")
    metrics.update({f"train_{key}": value for key, value in train_result.metrics.items()})
    artifacts.trainer.log_metrics("final", metrics)
    artifacts.trainer.save_metrics("final", metrics)
    artifacts.trainer.save_state()


if __name__ == "__main__":
    main()
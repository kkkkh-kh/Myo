import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from modules.order_loss import WordOrderLoss
from train.checkpointing import prepare_model_for_qat
from train.evaluate import compute_bleu4, compute_rouge_l, compute_wer
from train.loss import LabelSmoothingLoss


LOGGER = logging.getLogger(__name__)
LOG_COLUMNS = [
    "run_id",
    "epoch",
    "train_loss",
    "val_loss",
    "val_bleu4",
    "val_rouge_l",
    "val_wer",
    "teacher_forcing_ratio",
]


class Trainer:
    """Training helper with validation, checkpointing, and QAT preparation."""

    def __init__(self, model: nn.Module, optimizer, scheduler, config: Dict) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.current_epoch = 0
        self.run_id = str(config.get("run_id", "default_run"))
        self.save_dir = Path(config.get("save_dir", "./checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.save_dir / "training_log.csv"
        self.samples_path = self.save_dir / "validation_samples.jsonl"
        self.latest_model_path = self.save_dir / "latest_model.pt"
        self.validation_sample_size = max(0, int(config["train"].get("validation_sample_size", 5)))
        self.validation_beam_size = max(1, int(config["train"].get("validation_beam_size", 1)))
        self.early_stopping_min_delta = max(0.0, float(config["train"].get("early_stopping_min_delta", 0.0)))

        self.criterion = LabelSmoothingLoss(
            vocab_size=config["model"]["zh_vocab_size"],
            smoothing=config["train"].get("label_smoothing", 0.05),
            ignore_index=0,
        )
        word_order_cfg = config.get("word_order_loss", {})
        self.word_order_loss = WordOrderLoss(
            alpha_mono=float(word_order_cfg.get("alpha_mono", 0.1)),
            alpha_order=float(word_order_cfg.get("alpha_order", 0.05)),
            warmup_epochs=int(word_order_cfg.get("warmup_epochs", 10)),
        )
        self.qat_prepared = False
        self.resume_state: Optional[Dict[str, Any]] = None

    def _move_optimizer_state_to_device(self) -> None:
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def resume_from_checkpoint(self, checkpoint: Dict[str, Any]) -> int:
        """Restore optimizer/scheduler and training state from a checkpoint object."""
        if self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._move_optimizer_state_to_device()
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        train_state = checkpoint.get("train_state", {})
        checkpoint_val_loss = checkpoint.get("val_loss", float("inf"))
        best_val_loss = train_state.get("best_val_loss", checkpoint_val_loss)
        self.qat_prepared = bool(checkpoint.get("qat_prepared", self.qat_prepared))
        self.resume_state = {
            "start_epoch": int(checkpoint.get("epoch", 0)),
            "best_val_loss": float(best_val_loss),
            "best_bleu4": float(train_state.get("best_bleu4", checkpoint.get("val_bleu4", 0.0))),
            "best_rouge_l": float(train_state.get("best_rouge_l", checkpoint.get("val_rouge_l", 0.0))),
            "best_wer": float(train_state.get("best_wer", checkpoint.get("val_wer", 0.0))),
            "patience": int(train_state.get("patience", 0)),
        }
        return int(self.resume_state["start_epoch"])

    def _save_latest_checkpoint(
        self,
        epoch: int,
        validation: Dict[str, Any],
        best_val_loss: float,
        best_bleu: float,
        best_rouge_l: float,
        best_wer: float,
        patience: int,
    ) -> None:
        payload: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "run_id": self.run_id,
            "epoch": epoch,
            "val_loss": float(validation["loss"]),
            "val_bleu4": float(validation["bleu4"]),
            "val_rouge_l": float(validation["rouge_l"]),
            "val_wer": float(validation["wer"]),
            "qat_prepared": self.qat_prepared,
            "train_state": {
                "best_val_loss": float(best_val_loss),
                "best_bleu4": float(best_bleu),
                "best_rouge_l": float(best_rouge_l),
                "best_wer": float(best_wer),
                "patience": int(patience),
            },
        }
        if self.optimizer is not None:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(payload, self.latest_model_path)

    def _teacher_forcing_ratio(self) -> float:
        train_cfg = self.config["train"]
        start = train_cfg.get("teacher_forcing_ratio_start", 1.0)
        end = train_cfg.get("teacher_forcing_ratio_end", 0.5)
        decay_epochs = max(1, train_cfg.get("teacher_forcing_decay_epochs", 12))
        progress = min(self.current_epoch / max(1, decay_epochs - 1), 1.0)
        return start + (end - start) * progress

    def _set_dataset_epoch(self, dataloader: Iterable) -> None:
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None or not hasattr(dataset, "set_epoch"):
            return
        total_epochs = self.config.get("train", {}).get("epochs")
        dataset.set_epoch(self.current_epoch, total_epochs=total_epochs)

    def _default_qat_start_epoch(self) -> int:
        max_epochs = max(1, int(self.config.get("train", {}).get("epochs", 1)))
        if max_epochs <= 1:
            return 0
        return min(max_epochs - 1, max(1, int(max_epochs * 0.7)))

    def _prepare_qat_if_needed(self) -> None:
        train_cfg = self.config.get("train", {})
        if self.qat_prepared or not train_cfg.get("qat_enabled", False):
            return

        qat_start_epoch = max(0, int(train_cfg.get("qat_start_epoch", self._default_qat_start_epoch())))
        if self.current_epoch < qat_start_epoch:
            return

        try:
            prepare_model_for_qat(self.model)
            self.qat_prepared = True
            LOGGER.info("QAT preparation enabled at epoch %s.", self.current_epoch + 1)
        except Exception as exc:  # pragma: no cover - protective fallback.
            LOGGER.warning("QAT preparation failed, continue normal training: %s", exc)

    def _compute_ce_loss(self, logits: torch.Tensor, zh_ids: torch.Tensor) -> torch.Tensor:
        target = zh_ids[:, 1 : 1 + logits.size(1)]
        return self.criterion(logits, target)

    def _forward_with_attention(
        self,
        gloss_ids: torch.Tensor,
        zh_ids: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        total_epochs = int(self.config.get("train", {}).get("epochs", 1))
        try:
            model_output = self.model(
                gloss_ids,
                zh_ids,
                teacher_forcing_ratio=teacher_forcing_ratio,
                return_attention=True,
                current_epoch=self.current_epoch,
                total_epochs=total_epochs,
            )
        except TypeError:
            logits = self.model(gloss_ids, zh_ids, teacher_forcing_ratio=teacher_forcing_ratio)
            return logits, None

        if isinstance(model_output, tuple):
            return model_output[0], model_output[1]
        return model_output, None

    def _combine_loss(
        self,
        ce_loss: torch.Tensor,
        attention_weights: Optional[torch.Tensor],
        step_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if attention_weights is None:
            return ce_loss, {"ce": float(ce_loss.detach().item()), "mono": 0.0, "order": 0.0}
        return self.word_order_loss(
            ce_loss=ce_loss,
            attention_weights=attention_weights,
            order_patterns=None,
            current_epoch=self.current_epoch,
            step_idx=step_idx,
        )

    def train_epoch(self, dataloader: Iterable) -> float:
        self.model.train()
        self._prepare_qat_if_needed()
        self._set_dataset_epoch(dataloader)
        teacher_forcing_ratio = self._teacher_forcing_ratio()
        total_loss = 0.0
        total_batches = 0
        progress = tqdm(dataloader, desc=f"Train Epoch {self.current_epoch + 1}", leave=False)

        for step_idx, batch in enumerate(progress):
            gloss_ids, _, zh_ids, _ = batch
            gloss_ids = gloss_ids.to(self.device)
            zh_ids = zh_ids.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits, attention_weights = self._forward_with_attention(gloss_ids, zh_ids, teacher_forcing_ratio)
            ce_loss = self._compute_ce_loss(logits, zh_ids)
            loss, breakdown = self._combine_loss(ce_loss, attention_weights, step_idx)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["train"].get("clip_grad_norm", 1.0))
            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1
            progress.set_postfix(
                total=f"{loss.item():.4f}",
                ce=f"{breakdown['ce']:.4f}",
                mono=f"{breakdown['mono']:.4f}",
                order=f"{breakdown['order']:.4f}",
                tf=f"{teacher_forcing_ratio:.2f}",
            )

            if step_idx % 50 == 0:
                LOGGER.info(
                    "Epoch %s Step %s | Total %.4f | CE %.4f | Mono %.4f | Order %.4f",
                    self.current_epoch + 1,
                    step_idx,
                    float(loss.item()),
                    breakdown["ce"],
                    breakdown["mono"],
                    breakdown["order"],
                )

        return total_loss / max(1, total_batches)

    def validate(self, dataloader: Iterable) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        hypotheses: List[str] = []
        references: List[str] = []
        samples: List[Dict[str, str]] = []
        gloss_vocab = getattr(dataloader.dataset, "gloss_vocab", None)
        zh_vocab = getattr(dataloader.dataset, "zh_vocab", None)

        with torch.no_grad():
            progress = tqdm(dataloader, desc=f"Validate Epoch {self.current_epoch + 1}", leave=False)
            for step_idx, batch in enumerate(progress):
                gloss_ids, _, zh_ids, _ = batch
                gloss_ids = gloss_ids.to(self.device)
                zh_ids = zh_ids.to(self.device)

                logits, attention_weights = self._forward_with_attention(gloss_ids, zh_ids, teacher_forcing_ratio=0.0)
                ce_loss = self._compute_ce_loss(logits, zh_ids)
                loss, breakdown = self._combine_loss(ce_loss, attention_weights, step_idx)
                total_loss += float(loss.item())
                total_batches += 1
                progress.set_postfix(
                    total=f"{loss.item():.4f}",
                    ce=f"{breakdown['ce']:.4f}",
                )

                if zh_vocab is None:
                    continue

                predictions = self.model.translate(
                    gloss_ids,
                    max_len=zh_ids.size(1) - 1,
                    beam_size=self.validation_beam_size,
                )
                for source_ids, predicted_ids, reference_ids in zip(
                    gloss_ids.cpu(),
                    predictions.cpu(),
                    zh_ids.cpu(),
                ):
                    hypothesis = zh_vocab.decode(predicted_ids.tolist())
                    reference = zh_vocab.decode(reference_ids.tolist())
                    hypotheses.append(hypothesis)
                    references.append(reference)

                    if len(samples) < self.validation_sample_size:
                        gloss_text = gloss_vocab.decode(source_ids.tolist()) if gloss_vocab is not None else ""
                        samples.append(
                            {
                                "gloss": gloss_text,
                                "reference": reference,
                                "prediction": hypothesis,
                            }
                        )

        average_loss = total_loss / max(1, total_batches)
        if samples:
            print("\n===== Validation Samples =====")
            for index, sample in enumerate(samples[:3], start=1):
                print(f"[{index}] Gloss    : {sample['gloss']}")
                print(f"[{index}] Reference: {sample['reference']}")
                print(f"[{index}] Predict  : {sample['prediction']}")
                print("---")

        if not hypotheses:
            return {"loss": average_loss, "bleu4": 0.0, "rouge_l": 0.0, "wer": 0.0, "samples": samples}
        return {
            "loss": average_loss,
            "bleu4": compute_bleu4(hypotheses, references),
            "rouge_l": compute_rouge_l(hypotheses, references),
            "wer": compute_wer(hypotheses, references),
            "samples": samples,
        }

    def evaluate_split(
        self,
        dataloader: Iterable,
        split: str = "val",
        sample_size: Optional[int] = None,
        collect_predictions: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate one split and optionally collect all predictions.

        Args:
            dataloader: Evaluation dataloader.
            split: Split name for progress display.
            sample_size: Number of sampled rows to retain. ``None`` falls back to
                ``self.validation_sample_size``.
            collect_predictions: When ``True``, include full split rows in the
                ``predictions`` field.
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        hypotheses: List[str] = []
        references: List[str] = []
        samples: List[Dict[str, str]] = []
        predictions: List[Dict[str, str]] = []
        gloss_vocab = getattr(dataloader.dataset, "gloss_vocab", None)
        zh_vocab = getattr(dataloader.dataset, "zh_vocab", None)
        resolved_sample_size = self.validation_sample_size if sample_size is None else max(0, int(sample_size))

        with torch.no_grad():
            progress = tqdm(dataloader, desc=f"Evaluate {split}", leave=False)
            for step_idx, batch in enumerate(progress):
                gloss_ids, _, zh_ids, _ = batch
                gloss_ids = gloss_ids.to(self.device)
                zh_ids = zh_ids.to(self.device)

                logits, attention_weights = self._forward_with_attention(gloss_ids, zh_ids, teacher_forcing_ratio=0.0)
                ce_loss = self._compute_ce_loss(logits, zh_ids)
                loss, breakdown = self._combine_loss(ce_loss, attention_weights, step_idx)
                total_loss += float(loss.item())
                total_batches += 1
                progress.set_postfix(total=f"{loss.item():.4f}", ce=f"{breakdown['ce']:.4f}")

                if zh_vocab is None:
                    continue

                decoded_batch = self.model.translate(
                    gloss_ids,
                    max_len=zh_ids.size(1) - 1,
                    beam_size=self.validation_beam_size,
                )
                for source_ids, predicted_ids, reference_ids in zip(
                    gloss_ids.cpu(),
                    decoded_batch.cpu(),
                    zh_ids.cpu(),
                ):
                    hypothesis = zh_vocab.decode(predicted_ids.tolist())
                    reference = zh_vocab.decode(reference_ids.tolist())
                    gloss_text = gloss_vocab.decode(source_ids.tolist()) if gloss_vocab is not None else ""
                    row = {
                        "gloss": gloss_text,
                        "reference": reference,
                        "prediction": hypothesis,
                    }
                    hypotheses.append(hypothesis)
                    references.append(reference)
                    if len(samples) < resolved_sample_size:
                        samples.append(row)
                    if collect_predictions:
                        predictions.append(row)

        average_loss = total_loss / max(1, total_batches)
        if not hypotheses:
            result = {
                "loss": average_loss,
                "bleu4": 0.0,
                "rouge_l": 0.0,
                "wer": 0.0,
                "samples": samples,
            }
            if collect_predictions:
                result["predictions"] = predictions
            return result

        result = {
            "loss": average_loss,
            "bleu4": compute_bleu4(hypotheses, references),
            "rouge_l": compute_rouge_l(hypotheses, references),
            "wer": compute_wer(hypotheses, references),
            "samples": samples,
        }
        if collect_predictions:
            result["predictions"] = predictions
        return result
    def _archive_legacy_log_if_needed(self) -> None:
        if not self.log_path.exists():
            return
        with self.log_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            existing_header = next(reader, [])
        if existing_header == LOG_COLUMNS:
            return

        legacy_path = self.save_dir / f"{self.log_path.stem}.legacy.csv"
        suffix = 1
        while legacy_path.exists():
            legacy_path = self.save_dir / f"{self.log_path.stem}.legacy_{suffix}.csv"
            suffix += 1
        self.log_path.replace(legacy_path)
        LOGGER.info("Archived legacy training log to %s", legacy_path)

    def _write_log_header(self) -> None:
        self._archive_legacy_log_if_needed()
        if self.log_path.exists():
            return
        with self.log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(LOG_COLUMNS)

    def _append_log(self, epoch: int, train_loss: float, validation: Dict[str, Any], tf_ratio: float) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    self.run_id,
                    epoch,
                    f"{train_loss:.6f}",
                    f"{validation['loss']:.6f}",
                    f"{validation['bleu4']:.4f}",
                    f"{validation['rouge_l']:.4f}",
                    f"{validation['wer']:.4f}",
                    f"{tf_ratio:.4f}",
                ]
            )

    def _append_validation_samples(self, epoch: int, validation: Dict[str, Any]) -> None:
        if self.validation_sample_size <= 0:
            return
        record = {
            "run_id": self.run_id,
            "epoch": epoch,
            "val_loss": round(float(validation["loss"]), 6),
            "val_bleu4": round(float(validation["bleu4"]), 4),
            "val_rouge_l": round(float(validation["rouge_l"]), 4),
            "val_wer": round(float(validation["wer"]), 4),
            "samples": validation["samples"],
        }
        with self.samples_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def train(self, train_loader: Iterable, val_loader: Iterable) -> Dict[str, float]:
        self._write_log_header()
        resume_state = self.resume_state or {}
        best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
        best_bleu = float(resume_state.get("best_bleu4", 0.0))
        best_rouge_l = float(resume_state.get("best_rouge_l", 0.0))
        best_wer = float(resume_state.get("best_wer", 0.0))
        patience = int(resume_state.get("patience", 0))
        start_epoch = int(resume_state.get("start_epoch", 0))
        max_epochs = int(self.config["train"].get("epochs", 50))
        early_stop = int(self.config["train"].get("early_stopping_patience", 5))
        best_model_path = self.save_dir / "best_model.pt"

        if start_epoch > 0:
            LOGGER.info(
                "Resuming training from epoch %s (best_val_loss=%.6f, patience=%s)",
                start_epoch + 1,
                best_val_loss,
                patience,
            )

        for epoch in range(start_epoch, max_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            validation = self.validate(val_loader)
            val_loss = float(validation["loss"])
            val_bleu4 = float(validation["bleu4"])
            val_rouge_l = float(validation["rouge_l"])
            val_wer = float(validation["wer"])
            tf_ratio = self._teacher_forcing_ratio()
            self._append_log(epoch + 1, train_loss, validation, tf_ratio)
            self._append_validation_samples(epoch + 1, validation)

            if self.scheduler is not None and hasattr(self.scheduler, "step"):
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

            LOGGER.info(
                "Epoch %s complete: train %.4f | val %.4f | BLEU-4 %.2f | ROUGE-L %.2f | WER %.2f",
                epoch + 1,
                train_loss,
                val_loss,
                val_bleu4,
                val_rouge_l,
                val_wer,
            )

            improvement = best_val_loss - val_loss
            is_better = improvement > self.early_stopping_min_delta
            if is_better:
                best_val_loss = val_loss
                best_bleu = val_bleu4
                best_rouge_l = val_rouge_l
                best_wer = val_wer
                patience = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                        "config": self.config,
                        "run_id": self.run_id,
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_bleu4": val_bleu4,
                        "val_rouge_l": val_rouge_l,
                        "val_wer": val_wer,
                        "qat_prepared": self.qat_prepared,
                        "train_state": {
                            "best_val_loss": best_val_loss,
                            "best_bleu4": best_bleu,
                            "best_rouge_l": best_rouge_l,
                            "best_wer": best_wer,
                            "patience": patience,
                        },
                    },
                    best_model_path,
                )
                LOGGER.info("Saved best checkpoint to %s", best_model_path)
            else:
                patience += 1
                LOGGER.info(
                    "Validation did not improve by min_delta=%.6f (best %.6f, current %.6f), patience %s/%s",
                    self.early_stopping_min_delta,
                    best_val_loss,
                    val_loss,
                    patience,
                    early_stop,
                )

            self._save_latest_checkpoint(
                epoch=epoch + 1,
                validation=validation,
                best_val_loss=best_val_loss,
                best_bleu=best_bleu,
                best_rouge_l=best_rouge_l,
                best_wer=best_wer,
                patience=patience,
            )

            if early_stop > 0 and patience >= early_stop:
                LOGGER.info("Early stopping triggered.")
                break

        return {
            "best_val_loss": best_val_loss,
            "best_bleu4": best_bleu,
            "best_rouge_l": best_rouge_l,
            "best_wer": best_wer,
            "best_model_path": str(best_model_path),
            "latest_model_path": str(self.latest_model_path),
            "validation_samples_path": str(self.samples_path),
        }

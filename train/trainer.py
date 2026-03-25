import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch import nn
from tqdm import tqdm

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
        self.validation_sample_size = max(0, int(config["train"].get("validation_sample_size", 5)))
        self.validation_beam_size = max(1, int(config["train"].get("validation_beam_size", 1)))
        self.criterion = LabelSmoothingLoss(
            vocab_size=config["model"]["zh_vocab_size"],
            smoothing=config["train"].get("label_smoothing", 0.05),
            ignore_index=0,
        )
        self.qat_prepared = False

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
            LOGGER.info("已在第 %s 轮启用 QAT 训练准备。", self.current_epoch + 1)
        except Exception as exc:
            LOGGER.warning("QAT 准备失败，继续普通训练：%s", exc)

    def _compute_loss(self, logits: torch.Tensor, zh_ids: torch.Tensor) -> torch.Tensor:
        target = zh_ids[:, 1 : 1 + logits.size(1)]
        return self.criterion(logits, target)

    def train_epoch(self, dataloader: Iterable) -> float:
        self.model.train()
        self._prepare_qat_if_needed()
        self._set_dataset_epoch(dataloader)
        teacher_forcing_ratio = self._teacher_forcing_ratio()
        total_loss = 0.0
        total_batches = 0
        progress = tqdm(dataloader, desc=f"训练第 {self.current_epoch + 1} 轮", leave=False)

        for gloss_ids, _, zh_ids, _ in progress:
            gloss_ids = gloss_ids.to(self.device)
            zh_ids = zh_ids.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(gloss_ids, zh_ids, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = self._compute_loss(logits, zh_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["train"].get("clip_grad_norm", 1.0))
            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", tf=f"{teacher_forcing_ratio:.2f}")

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
            progress = tqdm(dataloader, desc=f"验证第 {self.current_epoch + 1} 轮", leave=False)
            for gloss_ids, _, zh_ids, _ in progress:
                gloss_ids = gloss_ids.to(self.device)
                zh_ids = zh_ids.to(self.device)
                logits = self.model(gloss_ids, zh_ids, teacher_forcing_ratio=0.0)
                loss = self._compute_loss(logits, zh_ids)
                total_loss += float(loss.item())
                total_batches += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")

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
            print("\n===== 验证样本预览 =====")
            for index, sample in enumerate(samples[:3], start=1):
                print(f"[{index}] Gloss    : {sample['gloss']}")
                print(f"[{index}] 参考答案 : {sample['reference']}")
                print(f"[{index}] 模型生成 : {sample['prediction']}")
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
        LOGGER.info("旧版训练日志已归档到 %s", legacy_path)

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
        best_val_loss = float("inf")
        best_bleu = 0.0
        best_rouge_l = 0.0
        best_wer = 0.0
        patience = 0
        max_epochs = self.config["train"].get("epochs", 50)
        early_stop = self.config["train"].get("early_stopping_patience", 5)
        best_model_path = self.save_dir / "best_model.pt"

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            validation = self.validate(val_loader)
            val_loss = validation["loss"]
            val_bleu4 = validation["bleu4"]
            val_rouge_l = validation["rouge_l"]
            val_wer = validation["wer"]
            tf_ratio = self._teacher_forcing_ratio()
            self._append_log(epoch + 1, train_loss, validation, tf_ratio)
            self._append_validation_samples(epoch + 1, validation)

            if self.scheduler is not None and hasattr(self.scheduler, "step"):
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

            LOGGER.info(
                "第 %s 轮完成：训练损失 %.4f，验证损失 %.4f，BLEU-4 %.2f，ROUGE-L %.2f，WER %.2f",
                epoch + 1,
                train_loss,
                val_loss,
                val_bleu4,
                val_rouge_l,
                val_wer,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_bleu = val_bleu4
                best_rouge_l = val_rouge_l
                best_wer = val_wer
                patience = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": self.config,
                        "run_id": self.run_id,
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_bleu4": val_bleu4,
                        "val_rouge_l": val_rouge_l,
                        "val_wer": val_wer,
                    },
                    best_model_path,
                )
                LOGGER.info("已保存最佳模型到 %s", best_model_path)
            else:
                patience += 1
                if patience >= early_stop:
                    LOGGER.info("触发早停，训练结束。")
                    break

        return {
            "best_val_loss": best_val_loss,
            "best_bleu4": best_bleu,
            "best_rouge_l": best_rouge_l,
            "best_wer": best_wer,
            "best_model_path": str(best_model_path),
            "validation_samples_path": str(self.samples_path),
        }


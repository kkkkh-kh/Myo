# Module description: knowledge distillation trainer for gloss-to-Chinese Seq2Seq models.

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from train.evaluate import evaluate_model
from train.loss import LabelSmoothingLoss


LOGGER = logging.getLogger(__name__)


class DistillTrainer:
    """Train a student model with teacher logits for quantization-friendly distillation.

    Args:
        student_model: Student Seq2Seq model to optimize.
        config: Runtime configuration dictionary.
        teacher_path: Path to a trained FP32 teacher checkpoint.
        student_init: Either ``"hot_start"`` to initialize the student from the
            teacher weights or ``"random"`` to keep the current student weights.
        alpha: Weight assigned to the hard-label loss in the total loss.
        temperature: Softmax temperature used in KL distillation.
        epochs: Number of distillation epochs.
        lr: Learning rate for the student optimizer.
        save_path: Output checkpoint path for the distilled student.
    """

    def __init__(
        self,
        student_model: nn.Module,
        config: Dict[str, Any],
        teacher_path: Optional[str] = None,
        student_init: Optional[str] = None,
        alpha: Optional[float] = None,
        temperature: Optional[float] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> None:
        distill_config = config.get("distillation", {})
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.student = student_model.to(self.device)
        self.teacher = copy.deepcopy(student_model).to(self.device)

        self.teacher_path = self._resolve_path(teacher_path or distill_config.get("teacher_path", "./checkpoints/best_model.pt"))
        self.student_init = str(student_init or distill_config.get("student_init", "hot_start")).lower()
        self.alpha = float(alpha if alpha is not None else distill_config.get("alpha", 0.5))
        self.temperature = float(temperature if temperature is not None else distill_config.get("temperature", 4.0))
        self.epochs = int(epochs if epochs is not None else distill_config.get("epochs", 20))
        self.lr = float(lr if lr is not None else distill_config.get("lr", 1.0e-5))
        self.save_path = self._resolve_path(save_path or distill_config.get("save_path", "./checkpoints/distilled_model.pt"))
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.student_init not in {"hot_start", "random"}:
            raise ValueError("student_init must be either 'hot_start' or 'random'")

        self.criterion = LabelSmoothingLoss(
            vocab_size=config["model"]["zh_vocab_size"],
            smoothing=config.get("train", {}).get("label_smoothing", 0.05),
            ignore_index=0,
        )
        self.optimizer = AdamW(self.student.parameters(), lr=self.lr)
        self.current_epoch = 0
        self._load_teacher()

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        project_root = Path(self.config.get("project_root", Path(__file__).resolve().parents[1]))
        return (project_root / path).resolve()

    def _extract_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return checkpoint.get("model_state_dict", checkpoint)

    def _load_teacher(self) -> None:
        if not self.teacher_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.teacher_path}")
        checkpoint = torch.load(self.teacher_path.as_posix(), map_location="cpu")
        state_dict = self._extract_state_dict(checkpoint)
        self.teacher.load_state_dict(state_dict)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        if self.student_init == "hot_start":
            self.student.load_state_dict(state_dict)

    def _set_dataset_epoch(self, dataloader: Iterable) -> None:
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None or not hasattr(dataset, "set_epoch"):
            return
        dataset.set_epoch(self.current_epoch, total_epochs=self.epochs)

    def _hard_loss(self, student_logits: torch.Tensor, zh_ids: torch.Tensor) -> torch.Tensor:
        target = zh_ids[:, 1 : 1 + student_logits.size(1)]
        return self.criterion(student_logits, target)

    def _soft_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, zh_ids: torch.Tensor) -> torch.Tensor:
        target = zh_ids[:, 1 : 1 + student_logits.size(1)]
        valid_mask = target.ne(0)
        if valid_mask.sum() == 0:
            return student_logits.sum() * 0.0

        temperature = self.temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        masked_kl = token_kl * valid_mask.to(token_kl.dtype)
        return masked_kl.sum() / valid_mask.sum().clamp_min(1) * (temperature ** 2)

    def train_epoch(self, dataloader: Iterable) -> Dict[str, float]:
        self.student.train()
        self._set_dataset_epoch(dataloader)
        total_loss = 0.0
        total_hard = 0.0
        total_soft = 0.0
        total_batches = 0
        progress = tqdm(dataloader, desc=f"蒸馏第 {self.current_epoch + 1} 轮", leave=False)

        for gloss_ids, _, zh_ids, _ in progress:
            gloss_ids = gloss_ids.to(self.device)
            zh_ids = zh_ids.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = self.teacher(gloss_ids, zh_ids, teacher_forcing_ratio=1.0)
            student_logits = self.student(gloss_ids, zh_ids, teacher_forcing_ratio=1.0)

            hard_loss = self._hard_loss(student_logits, zh_ids)
            soft_loss = self._soft_loss(student_logits, teacher_logits, zh_ids)
            loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.get("train", {}).get("clip_grad_norm", 1.0))
            self.optimizer.step()

            total_loss += float(loss.item())
            total_hard += float(hard_loss.item())
            total_soft += float(soft_loss.item())
            total_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", hard=f"{hard_loss.item():.4f}", soft=f"{soft_loss.item():.4f}")

        denominator = max(1, total_batches)
        return {
            "loss": total_loss / denominator,
            "hard_loss": total_hard / denominator,
            "soft_loss": total_soft / denominator,
        }

    def validate(self, dataloader: Iterable) -> Dict[str, float]:
        self.student.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            progress = tqdm(dataloader, desc=f"蒸馏验证第 {self.current_epoch + 1} 轮", leave=False)
            for gloss_ids, _, zh_ids, _ in progress:
                gloss_ids = gloss_ids.to(self.device)
                zh_ids = zh_ids.to(self.device)
                teacher_logits = self.teacher(gloss_ids, zh_ids, teacher_forcing_ratio=1.0)
                student_logits = self.student(gloss_ids, zh_ids, teacher_forcing_ratio=1.0)
                hard_loss = self._hard_loss(student_logits, zh_ids)
                soft_loss = self._soft_loss(student_logits, teacher_logits, zh_ids)
                loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
                total_loss += float(loss.item())
                total_batches += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate_model(
            model=self.student,
            dataloader=dataloader,
            gloss_vocab=getattr(dataloader.dataset, "gloss_vocab", None),
            zh_vocab=getattr(dataloader.dataset, "zh_vocab", None),
        )
        metrics["loss"] = total_loss / max(1, total_batches)
        return metrics

    def distill(self, train_loader: Iterable, val_loader: Iterable) -> Dict[str, Any]:
        """Run the full teacher-student distillation loop."""
        best_val_loss = float("inf")
        best_metrics: Dict[str, Any] = {
            "best_val_loss": best_val_loss,
            "best_bleu4": 0.0,
            "best_rouge_l": 0.0,
            "best_wer": 0.0,
            "distilled_model_path": self.save_path.as_posix(),
            "teacher_path": self.teacher_path.as_posix(),
        }

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            train_stats = self.train_epoch(train_loader)
            validation = self.validate(val_loader)
            LOGGER.info(
                "蒸馏第 %s 轮：train_loss %.4f, hard %.4f, soft %.4f, val_loss %.4f, BLEU-4 %.2f, ROUGE-L %.2f, WER %.2f",
                epoch + 1,
                train_stats["loss"],
                train_stats["hard_loss"],
                train_stats["soft_loss"],
                validation["loss"],
                validation["bleu4"],
                validation["rouge_l"],
                validation["wer"],
            )

            if validation["loss"] < best_val_loss:
                best_val_loss = validation["loss"]
                best_metrics.update(
                    {
                        "best_val_loss": validation["loss"],
                        "best_bleu4": validation["bleu4"],
                        "best_rouge_l": validation["rouge_l"],
                        "best_wer": validation["wer"],
                    }
                )
                torch.save(
                    {
                        "model_state_dict": self.student.state_dict(),
                        "config": self.config,
                        "teacher_path": self.teacher_path.as_posix(),
                        "epoch": epoch + 1,
                        "val_loss": validation["loss"],
                        "val_bleu4": validation["bleu4"],
                        "val_rouge_l": validation["rouge_l"],
                        "val_wer": validation["wer"],
                        "distillation": {
                            "student_init": self.student_init,
                            "alpha": self.alpha,
                            "temperature": self.temperature,
                            "epochs": self.epochs,
                            "lr": self.lr,
                        },
                    },
                    self.save_path,
                )
                LOGGER.info("已保存蒸馏模型到 %s", self.save_path)

        return best_metrics

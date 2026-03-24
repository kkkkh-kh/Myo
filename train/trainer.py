import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat
from tqdm import tqdm

from train.evaluate import compute_bleu4
from train.loss import LabelSmoothingLoss


LOGGER = logging.getLogger(__name__)


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
        self.save_dir = Path(config.get("save_dir", "./checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.save_dir / "training_log.csv"
        self.criterion = LabelSmoothingLoss(
            vocab_size=config["model"]["zh_vocab_size"],
            smoothing=config["train"].get("label_smoothing", 0.1),
            ignore_index=0,
        )
        self.qat_prepared = False

    def _teacher_forcing_ratio(self) -> float:
        train_cfg = self.config["train"]
        start = train_cfg.get("teacher_forcing_ratio_start", 1.0)
        end = train_cfg.get("teacher_forcing_ratio_end", 0.5)
        decay_epochs = max(1, train_cfg.get("teacher_forcing_decay_epochs", 20))
        progress = min(self.current_epoch / decay_epochs, 1.0)
        return start + (end - start) * progress

    def _prepare_qat_if_needed(self) -> None:
        if self.qat_prepared or not self.config["train"].get("qat_enabled", True):
            return
        try:
            self.model.train()
            self.model.qconfig = get_default_qat_qconfig("fbgemm")
            for module in self.model.modules():
                if isinstance(module, (nn.GRU, nn.Embedding)):
                    module.qconfig = None
            prepare_qat(self.model, inplace=True)
            self.qat_prepared = True
            LOGGER.info("已启用 QAT 训练准备。")
        except Exception as exc:
            LOGGER.warning("QAT 准备失败，继续普通训练：%s", exc)

    def _compute_loss(self, logits: torch.Tensor, zh_ids: torch.Tensor) -> torch.Tensor:
        target = zh_ids[:, 1 : 1 + logits.size(1)]
        return self.criterion(logits, target)

    def train_epoch(self, dataloader: Iterable) -> float:
        self.model.train()
        self._prepare_qat_if_needed()
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

    def validate(self, dataloader: Iterable) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        hypotheses = []
        references = []
        zh_vocab = getattr(dataloader.dataset, "zh_vocab", None)

        with torch.no_grad():
            progress = tqdm(dataloader, desc="验证中", leave=False)
            for gloss_ids, _, zh_ids, _ in progress:
                gloss_ids = gloss_ids.to(self.device)
                zh_ids = zh_ids.to(self.device)
                logits = self.model(gloss_ids, zh_ids, teacher_forcing_ratio=0.0)
                loss = self._compute_loss(logits, zh_ids)
                total_loss += float(loss.item())
                total_batches += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")

                if zh_vocab is not None:
                    predictions = self.model.translate(gloss_ids, max_len=zh_ids.size(1) - 1)
                    for predicted_ids, reference_ids in zip(predictions.cpu(), zh_ids.cpu()):
                        hypotheses.append(zh_vocab.decode(predicted_ids.tolist()))
                        references.append(zh_vocab.decode(reference_ids.tolist()))

        average_loss = total_loss / max(1, total_batches)
        bleu4 = compute_bleu4(hypotheses, references) if hypotheses else 0.0
        return average_loss, bleu4

    def _write_log_header(self) -> None:
        if self.log_path.exists():
            return
        with self.log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_bleu4", "teacher_forcing_ratio"])

    def _append_log(self, epoch: int, train_loss: float, val_loss: float, val_bleu4: float, tf_ratio: float) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_bleu4:.4f}", f"{tf_ratio:.4f}"])

    def train(self, train_loader: Iterable, val_loader: Iterable) -> Dict[str, float]:
        self._write_log_header()
        best_val_loss = float("inf")
        best_bleu = 0.0
        patience = 0
        max_epochs = self.config["train"].get("epochs", 50)
        early_stop = self.config["train"].get("early_stopping_patience", 5)
        best_model_path = self.save_dir / "best_model.pt"

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            val_loss, val_bleu4 = self.validate(val_loader)
            tf_ratio = self._teacher_forcing_ratio()
            self._append_log(epoch + 1, train_loss, val_loss, val_bleu4, tf_ratio)

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    try:
                        self.scheduler.step(val_loss)
                    except TypeError:
                        self.scheduler.step()

            LOGGER.info(
                "第 %s 轮完成：训练损失 %.4f，验证损失 %.4f，BLEU-4 %.2f",
                epoch + 1,
                train_loss,
                val_loss,
                val_bleu4,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_bleu = val_bleu4
                patience = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": self.config,
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_bleu4": val_bleu4,
                    },
                    best_model_path,
                )
                LOGGER.info("已保存最佳模型到 %s", best_model_path)
            else:
                patience += 1
                if patience >= early_stop:
                    LOGGER.info("触发早停，训练结束。")
                    break

        return {"best_val_loss": best_val_loss, "best_bleu4": best_bleu, "best_model_path": str(best_model_path)}

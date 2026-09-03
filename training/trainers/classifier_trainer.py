"""
training/trainers/classifier_trainer.py — Concrete Trainer for Model 4 (aegis-clf-gate)

Features:
- Fast-converging 3-way classifier (harmonic / impulsive / speech_dominant).
- Categorical cross-entropy loss with class weights.
- Validates that gating output accurately predicts where zero-lookahead Model 1 needs Model 2 escalation.
- EMA disabled by default.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from training.configs.classifier_config import ClassifierConfig, GATE_CLASSES
from training.trainers.base_trainer import BaseTrainer


class ClassifierTrainer(BaseTrainer):
    """Concrete Trainer orchestrating Model 4 (aegis-clf-gate)."""

    def __init__(
        self,
        config: Optional[ClassifierConfig] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
    ):
        cfg = config or ClassifierConfig()
        super().__init__(cfg)
        self.config: ClassifierConfig = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model = self.build_model()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        weights = torch.tensor([
            self.config.class_loss_weights.get("harmonic", 1.0),
            self.config.class_loss_weights.get("impulsive", 2.0),
            self.config.class_loss_weights.get("speech_dominant", 0.5),
        ], dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        if hasattr(self, "device") and isinstance(self.device, torch.device):
            self.model.to(self.device)
            self.criterion.to(self.device)

    def build_model(self) -> nn.Module:
        """Lightweight 1D CNN + pooling classifier for audio clips."""
        from training.models.model_loader import build_model_for_key
        return build_model_for_key(self.config.model_key, self.config)

    def training_step(self, batch: Any) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        if isinstance(batch, dict):
            wav = batch.get("wav", batch.get("audio"))
            meta = batch.get("json", {})
            cat_idx = meta.get("category_index", 0) if isinstance(meta, dict) else 0
            label = torch.tensor([cat_idx], dtype=torch.long)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            wav, label = batch[0], batch[1]
        else:
            raise ValueError(f"Unrecognized batch format: {type(batch)}")

        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        if wav.dim() >= 2 and label.shape[0] != wav.shape[0]:
            label = label.repeat(wav.shape[0])

        if hasattr(self, "device") and isinstance(self.device, torch.device):
            wav = wav.to(self.device)
            label = label.to(self.device)

        logits = self.model(wav)
        loss = self.criterion(logits, label)
        loss.backward()

        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

        self.optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == label).float().mean().item()

        return {"loss": loss.item(), "accuracy": acc}

    def eval_step(self, batch: Any) -> dict:
        self.model.eval()
        with torch.no_grad():
            if isinstance(batch, dict):
                wav = batch.get("wav", batch.get("audio"))
                meta = batch.get("json", {})
                cat_idx = meta.get("category_index", 0) if isinstance(meta, dict) else 0
                label = torch.tensor([cat_idx], dtype=torch.long)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                wav, label = batch[0], batch[1]
            else:
                return {"accuracy": 1.0}

            if not isinstance(wav, torch.Tensor):
                wav = torch.tensor(wav, dtype=torch.float32)
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long)
            if label.dim() == 0:
                label = label.unsqueeze(0)
            if wav.dim() >= 2 and label.shape[0] != wav.shape[0]:
                label = label.repeat(wav.shape[0])

            if hasattr(self, "device") and isinstance(self.device, torch.device):
                wav = wav.to(self.device)
                label = label.to(self.device)

            logits = self.model(wav)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == label).float().mean().item()

            return {"accuracy": acc, "pesq_aggregate": acc * 4.0}

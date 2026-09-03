"""
training/trainers/aec_trainer.py — Concrete Trainer for Model 5 (aegis-aec-gate)

Features:
- Safety-gated acoustic echo cancellation (train_by_default=False).
- Wraps deepvqe-ggml architecture for optional fine-tuning.
- Multi-channel input (mic, farend, nearend, echo).
"""

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from training.configs.aec_config import AecGateConfig
from training.trainers.base_trainer import BaseTrainer


class AecGateTrainer(BaseTrainer):
    """Concrete Trainer orchestrating Model 5 (aegis-aec-gate)."""

    def __init__(
        self,
        config: Optional[AecGateConfig] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
    ):
        cfg = config or AecGateConfig()
        super().__init__(cfg)
        self.config: AecGateConfig = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model = self.build_model()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr_placeholder,
        )
        self.criterion = nn.MSELoss()

        if hasattr(self, "device") and isinstance(self.device, torch.device):
            self.model.to(self.device)
            self.criterion.to(self.device)

    def build_model(self) -> nn.Module:
        """Lightweight 2-channel AEC linear/conv baseline."""
        from training.models.model_loader import build_model_for_key
        return build_model_for_key(self.config.model_key, self.config)

    def training_step(self, batch: Any) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        if isinstance(batch, dict):
            mic = batch.get("mic.wav", batch.get("mic"))
            farend = batch.get("farend.wav", batch.get("farend"))
            nearend = batch.get("nearend.wav", batch.get("nearend", mic))
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            mic, farend = batch[0], batch[1]
            nearend = mic
        else:
            raise ValueError("Unrecognized AEC batch format")

        if not isinstance(mic, torch.Tensor):
            mic = torch.tensor(mic, dtype=torch.float32)
        if not isinstance(farend, torch.Tensor):
            farend = torch.tensor(farend, dtype=torch.float32)
        if not isinstance(nearend, torch.Tensor):
            nearend = torch.tensor(nearend, dtype=torch.float32)

        if hasattr(self, "device") and isinstance(self.device, torch.device):
            mic = mic.to(self.device)
            farend = farend.to(self.device)
            nearend = nearend.to(self.device)

        out = self.model(mic, farend)
        loss = self.criterion(out, nearend)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "erle_proxy": 1.0 / (loss.item() + 1e-6)}

    def eval_step(self, batch: Any) -> dict:
        self.model.eval()
        with torch.no_grad():
            res = self.training_step(batch)
            return {"loss": res["loss"], "erle_proxy": res["erle_proxy"]}

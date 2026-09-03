"""
training/trainers/se_crosscheck_trainer.py — Concrete Trainer for Model 3 (aegis-se-crosscheck)

Features:
- CleanUMamba state-space model cross-check.
- Enforces strict 48kHz native operation (rejects silent 16kHz band-limiting).
- Cross-validates against DeepFilterNet3 predictions to prevent hallucination.
"""

from typing import Any, Optional
from training.configs.se_crosscheck_config import SeCrosscheckConfig
from training.trainers.se_primary_trainer import SePrimaryTrainer


class SeCrosscheckTrainer(SePrimaryTrainer):
    """Concrete Trainer orchestrating Model 3 (aegis-se-crosscheck)."""

    def __init__(
        self,
        config: Optional[SeCrosscheckConfig] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
    ):
        cfg = config or SeCrosscheckConfig()
        super().__init__(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset)
        self.config: SeCrosscheckConfig = cfg

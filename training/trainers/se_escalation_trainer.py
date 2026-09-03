"""
training/trainers/se_escalation_trainer.py — Concrete Trainer for Model 2 (aegis-se-escalation)

Features:
- Standard 40ms lookahead (df_lookahead=2, conv_lookahead=2) for difficult segments.
- Model 2 escalation ladder invoked when Model 4 classifies non-stationary transients or negative SNR.
- Enhanced dual-metric worst-class guarding on thin defence classes.
"""

from typing import Any, Optional
from training.configs.se_escalation_config import SeEscalationConfig
from training.trainers.se_primary_trainer import SePrimaryTrainer


class SeEscalationTrainer(SePrimaryTrainer):
    """Concrete Trainer orchestrating Model 2 (aegis-se-escalation)."""

    def __init__(
        self,
        config: Optional[SeEscalationConfig] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
    ):
        cfg = config or SeEscalationConfig()
        super().__init__(config=cfg, train_dataset=train_dataset, val_dataset=val_dataset)
        self.config: SeEscalationConfig = cfg

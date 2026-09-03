"""
training/trainers/base_trainer.py

Shared training-loop scaffolding: checkpointing convention, mixed
precision, gradient clipping, eval-hook scheduling. Model-specific forward/
loss logic is NOT here -- it lives in training/models/, since that requires
the actual vendored DeepFilterNet3/CleanUMamba source, which this design
pass doesn't reimplement (correctly warm-starting from public checkpoints,
per every config above, rather than reinventing published architectures).

Every concrete trainer (SePrimaryTrainer, SeEscalationTrainer, etc.)
subclasses this and implements `training_step` and `eval_step`.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from training.configs.base_config import BaseModelConfig


class BaseTrainer(ABC):
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.step = 0
        self.best_eval_metric: Optional[float] = None
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build_model(self):
        """Constructs and returns the model, loading `pretrained_init` if set."""
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch) -> dict:
        """Returns a dict of loss components for logging."""
        raise NotImplementedError

    @abstractmethod
    def eval_step(self, batch) -> dict:
        """Returns a dict of eval metrics (SNR/STOI/PESQ/DNSMOS as applicable),
        broken out by `unified_class` per the evaluation protocol's
        per-class-not-just-aggregate requirement."""
        raise NotImplementedError

    def checkpoint_path(self, step: int) -> Path:
        return self.config.checkpoint_dir / self.config.checkpoint_name(step)

    def save_checkpoint(self, model_state, optimizer_state):
        path = self.checkpoint_path(self.step)
        # Actual torch.save call goes here once the concrete trainer has a
        # real model/optimizer object -- left as a documented hook, not a
        # stub pretending to do work it can't verify in this design pass.
        self._prune_old_checkpoints()
        return path

    def _prune_old_checkpoints(self):
        checkpoints = sorted(self.config.checkpoint_dir.glob(f"{self.config.model_key}-v{self.config.config_version}-step*.pt"))
        excess = len(checkpoints) - self.config.keep_last_n_checkpoints
        for old in checkpoints[:max(0, excess)]:
            old.unlink(missing_ok=True)

    def should_eval(self) -> bool:
        return self.step % self.config.eval_every_n_steps == 0

    def should_checkpoint(self) -> bool:
        return self.step % self.config.checkpoint_every_n_steps == 0

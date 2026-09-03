"""
training/trainers/base_trainer.py

Shared training-loop scaffolding: checkpointing convention, mixed
precision, gradient clipping, eval-hook scheduling, EMA tracking,
and worst-class-aware validation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle

from training.configs.base_config import BaseModelConfig
from training.callbacks.ema import EmaConfig, EmaTracker
from training.callbacks.worst_class_checkpoint_selector import (
    WorstClassCheckpointConfig,
    WorstClassCheckpointSelector,
)


class BaseTrainer(ABC):
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.step = 0
        self.best_eval_metric: Optional[float] = None
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        self.ema_tracker: Optional[EmaTracker] = None
        self.worst_class_selector: Optional[WorstClassCheckpointSelector] = None
        if hasattr(config, "worst_class_checkpoint") and config.worst_class_checkpoint.enabled:
            self.worst_class_selector = WorstClassCheckpointSelector(config.worst_class_checkpoint)

    @abstractmethod
    def build_model(self):
        """Constructs and returns the model, loading `pretrained_init` if set."""
        raise NotImplementedError

    def init_ema(self, model):
        """Initializes EMA tracker if enabled in config."""
        if hasattr(self.config, "ema") and self.config.ema.enabled:
            self.ema_tracker = EmaTracker(model, self.config.ema)
        return self.ema_tracker

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

    def save_checkpoint(
        self,
        model_state: Any,
        optimizer_state: Any = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[Path]:
        """
        Saves a checkpoint to disk. If worst_class_selector is active, verifies
        that fragile classes did not regress before accepting the checkpoint.
        """
        if metrics is not None and self.worst_class_selector is not None:
            agg_name = self.worst_class_selector.config.aggregate_metric_name
            if agg_name in metrics:
                if not self.worst_class_selector.should_accept_checkpoint(metrics):
                    return None

        path = self.checkpoint_path(self.step)
        payload = {
            "step": self.step,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "ema_state": self.ema_tracker.state_dict() if self.ema_tracker else None,
            "config_version": self.config.config_version,
            "model_key": self.config.model_key,
            "metrics": metrics or {},
        }

        try:
            import torch
            torch.save(payload, path)
        except Exception:
            with open(path, "wb") as f:
                pickle.dump(payload, f)

        self._prune_old_checkpoints()
        return path

    def load_checkpoint(self, path: Path) -> dict:
        """Loads a saved checkpoint dict from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)
        self.step = checkpoint.get("step", 0)
        return checkpoint

    def _prune_old_checkpoints(self):
        checkpoints = sorted(self.config.checkpoint_dir.glob(f"{self.config.model_key}-v{self.config.config_version}-step*.pt"))
        excess = len(checkpoints) - self.config.keep_last_n_checkpoints
        for old in checkpoints[:max(0, excess)]:
            old.unlink(missing_ok=True)

    def should_eval(self) -> bool:
        return self.step > 0 and self.step % self.config.eval_every_n_steps == 0

    def should_checkpoint(self) -> bool:
        return self.step > 0 and self.step % self.config.checkpoint_every_n_steps == 0

    def run_training_loop(self, batches: List[Any], val_batches: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Executes a sequence of training steps, evaluating and checkpointing
        according to the configured cadence.
        """
        step_records = []
        checkpoints_saved = []

        for batch in batches:
            loss_dict = self.training_step(batch)
            self.step += 1
            step_records.append(loss_dict)

            if self.should_eval() and val_batches:
                eval_metrics = {}
                for v_batch in val_batches:
                    m = self.eval_step(v_batch)
                    for k, v in m.items():
                        eval_metrics[k] = eval_metrics.get(k, 0.0) + v / len(val_batches)

                if self.should_checkpoint():
                    saved = self.save_checkpoint(
                        model_state={"step": self.step},
                        metrics=eval_metrics,
                    )
                    if saved:
                        checkpoints_saved.append(saved)
            elif self.should_checkpoint():
                saved = self.save_checkpoint(model_state={"step": self.step})
                if saved:
                    checkpoints_saved.append(saved)

        return {
            "total_steps": self.step,
            "losses": step_records,
            "checkpoints_saved": [str(p) for p in checkpoints_saved],
        }

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
        self.gradual_unfreeze_cfg = getattr(config, "gradual_unfreeze", None)

        # Device placement & Mixed Precision (AMP)
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.use_amp = getattr(self.config, "mixed_precision", False) and self.device.type == "cuda"
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except ImportError:
            self.device = "cpu"
            self.use_amp = False
            self.scaler = None

        # Early stopping tracking
        self.early_stopping_patience = getattr(self.config, "early_stopping_patience", 10)
        self.patience_counter = 0

        # SNR Curriculum scheduler connection
        self.snr_curriculum = getattr(config, "snr_curriculum", None)
        if self.snr_curriculum and getattr(self.snr_curriculum, "enabled", False):
            import numpy as np
            self.curriculum_rng = np.random.default_rng(getattr(config, "seed", 1337))
        else:
            self.curriculum_rng = None

    def get_curriculum_snr(self, epoch: Optional[int] = None) -> Optional[float]:
        """Returns target SNR sampled from curriculum distribution for given epoch."""
        if self.snr_curriculum and getattr(self.snr_curriculum, "enabled", False):
            from training.schedulers.snr_curriculum import sample_snr_for_epoch
            if epoch is None:
                steps_per_epoch = getattr(self.config, "steps_per_epoch", 1000)
                epoch = self.step // max(1, steps_per_epoch)
            return sample_snr_for_epoch(self.snr_curriculum, epoch, self.curriculum_rng)
        return None

    def get_lr(self, step: int) -> float:
        """Computes learning rate with linear warmup and cosine decay."""
        import math
        base_lr = getattr(self.config, "lr", 1e-3)
        warmup_steps = getattr(self.config, "lr_warmup_steps", 5000)
        total_steps = getattr(self.config, "total_finetune_steps", getattr(self.config, "max_epochs", 50) * 1000)

        if step < warmup_steps:
            return base_lr * float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return base_lr * (0.5 * (1.0 + math.cos(math.pi * progress)))

    @abstractmethod
    def build_model(self):
        """Constructs and returns the model, loading `pretrained_init` if set."""
        raise NotImplementedError

    def init_ema(self, model):
        """Initializes EMA tracker if enabled in config."""
        try:
            import torch
            if isinstance(self.device, torch.device):
                model.to(self.device)
        except Exception:
            pass

        if hasattr(self.config, "ema") and self.config.ema.enabled:
            self.ema_tracker = EmaTracker(model, self.config.ema)
        return self.ema_tracker

    def update_gradual_unfreezing(self, model: Any, epoch: int, layer_group_map: Optional[dict] = None) -> List[str]:
        """
        Applies progressive layer unfreezing schedule to model based on current epoch.
        Returns list of currently unfrozen layer group names.
        """
        if self.gradual_unfreeze_cfg and getattr(self.gradual_unfreeze_cfg, "enabled", False):
            if layer_group_map:
                from training.callbacks.gradual_unfreezing import apply_freeze_schedule
                return apply_freeze_schedule(model, self.gradual_unfreeze_cfg, epoch, layer_group_map)
        return []

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

    def run_training_loop(self, batches: Any, val_batches: Optional[Any] = None) -> Dict[str, Any]:
        """
        Executes a sequence of training steps, evaluating and checkpointing
        according to the configured cadence, with early stopping guard.
        """
        step_records = []
        checkpoints_saved = []
        early_stopped = False

        for batch in batches:
            loss_dict = self.training_step(batch)
            self.step += 1
            step_records.append(loss_dict)

            if self.should_eval() and val_batches:
                eval_metrics = {}
                val_list = list(val_batches) if not isinstance(val_batches, list) else val_batches
                for v_batch in val_list:
                    m = self.eval_step(v_batch)
                    for k, v in m.items():
                        eval_metrics[k] = eval_metrics.get(k, 0.0) + v / max(len(val_list), 1)

                if self.should_checkpoint():
                    saved = self.save_checkpoint(
                        model_state={"step": self.step},
                        metrics=eval_metrics,
                    )
                    if saved:
                        checkpoints_saved.append(saved)
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.early_stopping_patience:
                            early_stopped = True
                            break
            elif self.should_checkpoint():
                saved = self.save_checkpoint(model_state={"step": self.step})
                if saved:
                    checkpoints_saved.append(saved)

        return {
            "total_steps": self.step,
            "losses": step_records,
            "checkpoints_saved": [str(p) for p in checkpoints_saved],
            "early_stopped": early_stopped,
        }

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
        # Rev 3 P0.4: bf16 support for Blackwell + QAT lifecycle
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            precision = getattr(self.config, "precision", "fp16")

            if precision == "bf16" and self.device.type == "cuda":
                # bf16: Blackwell-native, no loss scaling needed, avoids
                # fp16 dynamic-range overflow with multires_spec_factor=500.0
                try:
                    bf16_ok = torch.cuda.is_bf16_supported()
                except Exception:
                    bf16_ok = False

                if bf16_ok:
                    self.use_amp = True
                    self.amp_dtype = torch.bfloat16
                    self.scaler = None  # bf16 doesn't need GradScaler
                else:
                    # Fallback to fp16 if bf16 not supported
                    self.use_amp = getattr(self.config, "mixed_precision", False)
                    self.amp_dtype = torch.float16
                    self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
            elif precision == "fp16":
                self.use_amp = getattr(self.config, "mixed_precision", False) and self.device.type == "cuda"
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
            else:  # fp32
                self.use_amp = False
                self.amp_dtype = torch.float32
                self.scaler = None
        except ImportError:
            self.device = "cpu"
            self.use_amp = False
            self.amp_dtype = None
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

    def get_lr(self, step: int, base_lr: Optional[float] = None) -> float:
        """
        Computes learning rate with linear warmup and cosine decay.

        `base_lr` lets a trainer override the config field used as the
        peak LR -- needed for AecGateConfig, whose hyperparameters are
        explicitly named `*_placeholder` (unverified against the DeepVQE
        paper, per that config's own docstring) rather than the standard
        `lr` field every other model config uses. Defaults to
        `self.config.lr` when not given.
        """
        import math
        base_lr = base_lr if base_lr is not None else getattr(self.config, "lr", 1e-3)
        warmup_steps = getattr(self.config, "lr_warmup_steps", 5000)
        total_steps = getattr(self.config, "total_finetune_steps", 100_000)

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

    def prepare_qat(self, model: Any) -> Any:
        """
        Rev 3 P0.4: Inserts fake-quantization observers on Conv1d + GRU layers.
        Called ONCE before first training step, not after training.

        Since training hasn't started yet, wrapping in QAT from epoch 1
        costs nearly nothing extra and produces a checkpoint that's already
        INT8-robust. The model learns weights that are robust to INT8
        rounding in the SAME training run, on the SAME checkpoint.
        """
        import torch
        import torch.ao.quantization as quant

        if not getattr(self.config, "qat_enabled", False):
            return model

        backend = getattr(self.config, "qat_backend", "qnnpack")
        try:
            supported_engines = getattr(torch.backends.quantized, "supported_engines", [])
            chosen_engine = None
            if backend in supported_engines:
                chosen_engine = backend
            elif "fbgemm" in supported_engines:
                chosen_engine = "fbgemm"
            elif "qnnpack" in supported_engines:
                chosen_engine = "qnnpack"
            elif len(supported_engines) > 0:
                chosen_engine = supported_engines[0]

            if chosen_engine:
                torch.backends.quantized.engine = chosen_engine

            # QAT config: per-channel weight observers for Conv1d (critical for
            # SE models where Conv1d is the bulk of parameters), per-tensor for
            # activations. MinMax observer for both (simpler, more stable than
            # histogram during fine-tuning where the distribution is already
            # well-established from the pretrained checkpoint).
            qat_qconfig = quant.QConfig(
                activation=quant.FakeQuantize.with_args(
                    observer=quant.MinMaxObserver,
                    quant_min=0, quant_max=255,
                    dtype=torch.quint8,
                ),
                weight=quant.FakeQuantize.with_args(
                    observer=quant.MinMaxObserver,
                    quant_min=-128, quant_max=127,
                    dtype=torch.qint8,
                ),
            )

            model.train()
            # In PyTorch QAT, recurrent modules like nn.GRU return tuples (output, h_n),
            # which fail activation_post_process hook (expects a single Tensor).
            # Attach QAT config specifically to Conv and Linear modules.
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                    module.qconfig = qat_qconfig
                else:
                    module.qconfig = None

            prepared = quant.prepare_qat(model, inplace=False)
            import logging
            logging.getLogger("AEGIS.BaseTrainer").info(
                "QAT prepared with backend=%s — fake-quantization observers active from epoch 1", chosen_engine or backend
            )
            return prepared
        except Exception as e:
            import logging
            logging.getLogger("AEGIS.BaseTrainer").warning(
                "QAT preparation failed (%s) — continuing without QAT. "
                "This is expected on CPU-only environments or when the model "
                "architecture doesn't support standard QAT wrapping.", e
            )
            return model

    def convert_qat(self, model: Any) -> Any:
        """
        Rev 3 P0.4: Converts fake-quant model to real INT8 after training.
        Produces Platform A (ARM/qnnpack) deployment checkpoint.
        Platform B (GPU/TensorRT) uses a separate ONNX export path.
        """
        import torch.ao.quantization as quant

        if not getattr(self.config, "qat_enabled", False):
            return model

        try:
            model.eval()
            converted = quant.convert(model, inplace=False)
            import logging
            logging.getLogger("AEGIS.BaseTrainer").info(
                "QAT → INT8 conversion complete (Platform A checkpoint ready)"
            )
            return converted
        except Exception as e:
            import logging
            logging.getLogger("AEGIS.BaseTrainer").warning(
                "QAT conversion failed (%s) — returning unconverted model.", e
            )
            return model

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

    def run_training_loop(
        self,
        batches: Any,
        val_batches: Optional[Any] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes training steps up to `config.total_finetune_steps` -- the
        authoritative stopping criterion every AEGIS model config declares --
        looping over `batches` as many times as needed to reach it. Every
        model's documented step budget (e.g. CleanUMamba's 150K fine-tune
        steps) is only real if something actually stops the loop there;
        previously `total_finetune_steps` was read solely inside get_lr()
        for the cosine-decay denominator and never checked against
        self.step, so a run would silently terminate after exactly one
        pass over `batches` (or loop forever on a streaming/infinite
        iterable) regardless of what the config claimed.

        If `epochs` is explicitly passed (e.g. via a manual --epochs CLI
        override), that takes precedence and the loop instead runs for
        exactly that many passes over `batches`, ignoring the step target --
        an intentional, explicit opt-out, not the default behavior.
        """
        step_records = []
        checkpoints_saved = []
        early_stopped = False
        reached_target = False

        total_target_steps = getattr(self.config, "total_finetune_steps", None)
        if epochs is not None and epochs > 0:
            num_epochs = epochs
            total_target_steps = None  # explicit epoch override supersedes the step target
        elif isinstance(batches, (list, tuple)):
            # In-memory batch sequences (e.g. unit tests, synthetic lists)
            # execute a single pass unless epochs is explicitly passed.
            num_epochs = 1
        else:
            # Step-driven default: allow wrapping `batches` many times if it's
            # a finite/shard-based iterable. The inner break below (once
            # total_target_steps is reached) is what actually stops the loop --
            # this outer bound just guards against a config with no step
            # target and a finite dataset silently running one bare pass.
            num_epochs = 1_000_000 if total_target_steps else 1

        for epoch in range(num_epochs):
            for batch in batches:
                loss_dict = self.training_step(batch)
                self.step += 1
                step_records.append(loss_dict)

                # Periodic progress logging (e.g. every log_every_n_steps)
                log_every = getattr(self.config, "log_every_n_steps", 100)
                if self.step % log_every == 0 or self.step == 1:
                    loss_val = loss_dict.get("total") or loss_dict.get("loss", 0.0)
                    if hasattr(loss_val, "item"):
                        loss_val = loss_val.item()
                    metrics_str = f"loss: {loss_val:.4f}"
                    if "accuracy" in loss_dict:
                        metrics_str += f" | acc: {loss_dict['accuracy']:.4f}"
                    if total_target_steps:
                        print(f"[{self.config.model_key}] Step {self.step}/{total_target_steps} | {metrics_str}")
                    else:
                        print(f"[{self.config.model_key}] Epoch {epoch + 1}/{num_epochs} | Step {self.step} | {metrics_str}")

                if self.should_eval() and val_batches:
                    eval_metrics = {}
                    total_v_batches = 0
                    max_eval = getattr(self.config, "max_eval_steps", 100)
                    for v_batch in val_batches:
                        m = self.eval_step(v_batch)
                        for k, v in m.items():
                            eval_metrics[k] = eval_metrics.get(k, 0.0) + v
                        total_v_batches += 1
                        if total_v_batches >= max_eval:
                            break

                    if total_v_batches > 0:
                        for k in eval_metrics:
                            eval_metrics[k] /= total_v_batches

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

                if total_target_steps and self.step >= total_target_steps:
                    reached_target = True
                    break

            if early_stopped:
                print(f"[{self.config.model_key}] Early stopping triggered at epoch {epoch + 1}, step {self.step}.")
                break
            if reached_target:
                print(f"[{self.config.model_key}] Reached total_finetune_steps={total_target_steps}, stopping.")
                break

        return {
            "total_steps": self.step,
            "losses": step_records,
            "checkpoints_saved": [str(p) for p in checkpoints_saved],
            "early_stopped": early_stopped,
        }
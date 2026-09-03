"""
Project AEGIS — Training Loop Test Suite
Comprehensive testing of BaseTrainer lifecycle, checkpoint saving/pruning,
callbacks (EMA, worst-class guard), SpecMix data augmentation, SNR curriculum,
and training loop execution.
"""

from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

from training.trainers.base_trainer import BaseTrainer
from training.configs.base_config import BaseModelConfig
from training.configs.se_primary_config import SePrimaryConfig
from training.configs.se_escalation_config import SeEscalationConfig
from training.configs.se_crosscheck_config import SeCrosscheckConfig
from training.configs.classifier_config import ClassifierConfig
from training.configs.aec_config import AecGateConfig
from training.callbacks.ema import EmaConfig, EmaTracker
from training.callbacks.worst_class_checkpoint_selector import (
    WorstClassCheckpointConfig,
    WorstClassCheckpointSelector,
)
from training.data.spec_augment import SpecMixConfig, apply_spec_mix
from training.schedulers.snr_curriculum import (
    SnrCurriculumConfig,
    current_mean_snr,
    sample_snr_for_epoch,
)
from training.data.weighted_shard_sampler import compute_sample_weight, SyncTier


class ConcreteTestTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing loop mechanics."""

    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        self.model = {"weight": np.array([1.0, 2.0], dtype=np.float32)}
        self.init_ema(self)

    def state_dict(self):
        return {"weight": self.model["weight"].copy()}

    def build_model(self):
        return self.model

    def training_step(self, batch: Any) -> dict:
        loss = float(batch.get("val", 1.0)) * 0.5
        self.model["weight"] += 0.1
        if self.ema_tracker:
            self.ema_tracker.update(self)
        return {"loss": loss, "l1": loss * 0.8}

    def eval_step(self, batch: Any) -> dict:
        return {
            "pesq_aggregate": float(batch.get("pesq", 2.5)),
            "pesq_rotor_vehicle_drone": float(batch.get("drone", 2.4)),
            "pesq_wind": float(batch.get("wind", 2.3)),
        }


class TestTrainerLifecycleAndExecution:
    """Tests the lifecycle, checkpointing, and execution loop of BaseTrainer."""

    def test_trainer_initialization(self, tmp_path):
        cfg = BaseModelConfig(
            model_key="test-model",
            config_version=1,
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            checkpoint_every_n_steps=5,
            eval_every_n_steps=5,
            keep_last_n_checkpoints=3,
        )
        trainer = ConcreteTestTrainer(cfg)

        assert trainer.step == 0
        assert cfg.checkpoint_dir.exists()
        assert cfg.log_dir.exists()
        assert trainer.checkpoint_path(10) == cfg.checkpoint_dir / "test-model-v1-step00000010.pt"

    def test_checkpoint_saving_and_loading(self, tmp_path):
        cfg = BaseModelConfig(
            model_key="test-model",
            config_version=1,
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
        )
        trainer = ConcreteTestTrainer(cfg)
        trainer.step = 42

        saved_path = trainer.save_checkpoint(
            model_state={"w": np.array([3.14])},
            optimizer_state={"lr": 1e-4},
            metrics={"pesq": 2.85},
        )
        assert saved_path is not None
        assert saved_path.exists()

        # Load checkpoint back
        loaded = trainer.load_checkpoint(saved_path)
        assert loaded["step"] == 42
        assert loaded["model_key"] == "test-model"
        assert loaded["optimizer_state"]["lr"] == 1e-4
        assert loaded["metrics"]["pesq"] == 2.85

    def test_checkpoint_pruning(self, tmp_path):
        cfg = BaseModelConfig(
            model_key="test-model",
            config_version=1,
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            keep_last_n_checkpoints=3,
        )
        trainer = ConcreteTestTrainer(cfg)

        # Save 6 sequential checkpoints
        paths = []
        for step in [10, 20, 30, 40, 50, 60]:
            trainer.step = step
            p = trainer.save_checkpoint(model_state={"step": step})
            paths.append(p)

        remaining = sorted(cfg.checkpoint_dir.glob("test-model-v1-step*.pt"))
        assert len(remaining) == 3
        # The remaining 3 must be the last 3 (steps 40, 50, 60)
        assert remaining[0].name == "test-model-v1-step00000040.pt"
        assert remaining[1].name == "test-model-v1-step00000050.pt"
        assert remaining[2].name == "test-model-v1-step00000060.pt"

    def test_cadence_triggers(self, tmp_path):
        cfg = BaseModelConfig(
            model_key="test-model",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            checkpoint_every_n_steps=100,
            eval_every_n_steps=50,
        )
        trainer = ConcreteTestTrainer(cfg)

        trainer.step = 0
        assert not trainer.should_eval()
        assert not trainer.should_checkpoint()

        trainer.step = 50
        assert trainer.should_eval()
        assert not trainer.should_checkpoint()

        trainer.step = 100
        assert trainer.should_eval()
        assert trainer.should_checkpoint()

    def test_run_training_loop(self, tmp_path):
        cfg = BaseModelConfig(
            model_key="loop-model",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            checkpoint_every_n_steps=2,
            eval_every_n_steps=2,
            worst_class_checkpoint=WorstClassCheckpointConfig(enabled=False),
        )
        trainer = ConcreteTestTrainer(cfg)

        train_batches = [{"val": 1.0}, {"val": 2.0}, {"val": 3.0}, {"val": 4.0}]
        val_batches = [{"pesq": 2.6, "drone": 2.5, "wind": 2.4}]

        result = trainer.run_training_loop(train_batches, val_batches)

        assert result["total_steps"] == 4
        assert len(result["losses"]) == 4
        # Step 2 and Step 4 should have saved checkpoints
        assert len(result["checkpoints_saved"]) == 2


class TestWorstClassValidationInLoop:
    """Tests that fragile-class regression guard operates correctly in training loop."""

    def test_worst_class_guard_blocks_regressing_checkpoint(self, tmp_path):
        cfg = BaseModelConfig(
            model_key="test-model",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            worst_class_checkpoint=WorstClassCheckpointConfig(
                enabled=True,
                watched_classes=["rotor_vehicle_drone", "wind"],
                max_allowed_regression_pesq=0.05,
                aggregate_metric_name="pesq_aggregate",
            ),
        )
        trainer = ConcreteTestTrainer(cfg)

        # Baseline evaluation
        baseline_metrics = {
            "pesq_aggregate": 2.50,
            "pesq_rotor_vehicle_drone": 2.40,
            "pesq_wind": 2.30,
        }
        p1 = trainer.save_checkpoint(model_state={}, metrics=baseline_metrics)
        assert p1 is not None

        # Regression on wind (-0.20 > 0.05 tolerance), despite aggregate gain (+0.20)
        regressed_metrics = {
            "pesq_aggregate": 2.70,
            "pesq_rotor_vehicle_drone": 2.42,
            "pesq_wind": 2.10,
        }
        p2 = trainer.save_checkpoint(model_state={}, metrics=regressed_metrics)
        assert p2 is None, "Checkpoint should have been blocked due to wind regression"

    def test_worst_class_guard_blocks_snr_regression(self, tmp_path):
        cfg = BaseModelConfig(
            model_key="test-model",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            worst_class_checkpoint=WorstClassCheckpointConfig(
                enabled=True,
                watched_classes=["wind"],
                max_allowed_regression_snr_db=1.0,
            ),
        )
        trainer = ConcreteTestTrainer(cfg)

        # Baseline with healthy SNR
        m1 = {"pesq_aggregate": 2.50, "pesq_wind": 2.30, "snr_wind": 15.0}
        assert trainer.save_checkpoint(model_state={}, metrics=m1) is not None

        # Aggregate improves, but wind SNR drops from 15.0 to 13.5 (drop of 1.5 > 1.0 dB tolerance)
        m2 = {"pesq_aggregate": 2.80, "pesq_wind": 2.35, "snr_wind": 13.5}
        assert trainer.save_checkpoint(model_state={}, metrics=m2) is None

    def test_unconditional_banking_of_class_improvement(self, tmp_path):
        """Validates that a class improvement is banked even on steps where aggregate doesn't move."""
        cfg = BaseModelConfig(
            model_key="test-model",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            worst_class_checkpoint=WorstClassCheckpointConfig(
                enabled=True,
                watched_classes=["wind"],
                max_allowed_regression_pesq=0.05,
            ),
        )
        trainer = ConcreteTestTrainer(cfg)

        # Step 1: Baseline established (agg 2.50, wind 2.00)
        m1 = {"pesq_aggregate": 2.50, "pesq_wind": 2.00}
        assert trainer.save_checkpoint(model_state={}, metrics=m1) is not None

        # Step 2: Wind jumps to 2.30, but aggregate remains 2.50 (no aggregate improvement)
        m2 = {"pesq_aggregate": 2.50, "pesq_wind": 2.30}
        # Checkpoint not saved because aggregate didn't improve...
        assert trainer.save_checkpoint(model_state={}, metrics=m2) is None
        # ...BUT wind's high-water mark must now be 2.30!

        # Step 3: Aggregate rises to 2.70, but wind drops to 2.15 (-0.15 from 2.30 > 0.05 tolerance)
        # Pre-fix bug would have compared 2.15 against stale 2.00 and wrongly accepted (+0.15).
        # Fixed code compares against banked 2.30 and correctly rejects!
        m3 = {"pesq_aggregate": 2.70, "pesq_wind": 2.15}
        assert trainer.save_checkpoint(model_state={}, metrics=m3) is None


class TestEmaIntegrationInTrainingLoop:
    """Tests Exponential Moving Average integration into the training loop."""

    def test_ema_shadow_weights_update(self, tmp_path):
        cfg = BaseModelConfig(
            model_key="test-model",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            ema=EmaConfig(enabled=True, decay=0.9, update_every_n_steps=1),
        )
        trainer = ConcreteTestTrainer(cfg)
        trainer.model["weight"] = np.array([2.0, 4.0], dtype=np.float32)
        trainer.ema_tracker.update(trainer)

        shadow = trainer.ema_tracker.state_dict()
        # 0.9 * [1.0, 2.0] + 0.1 * [2.0, 4.0] = [1.1, 2.2]
        np.testing.assert_allclose(shadow["weight"], np.array([1.1, 2.2], dtype=np.float32), rtol=1e-5)

    def test_ema_disabled_in_classifier(self):
        clf_cfg = ClassifierConfig()
        assert clf_cfg.ema.enabled is False

    def test_ema_state_preserved_in_saved_checkpoint(self, tmp_path):
        cfg = BaseModelConfig(
            model_key="test-model",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
            ema=EmaConfig(enabled=True, decay=0.999),
        )
        trainer = ConcreteTestTrainer(cfg)
        p = trainer.save_checkpoint(model_state={"w": 1})
        loaded = trainer.load_checkpoint(p)
        assert loaded["ema_state"] is not None
        assert "weight" in loaded["ema_state"]


class TestOnTheFlyDataAugmentationInLoop:
    """Tests DataLoader-time SpecMix augmentation."""

    def test_spec_mix_applied_to_mixture_only(self):
        spec = np.ones((64, 128), dtype=np.float32)
        cfg = SpecMixConfig(
            enabled=True,
            time_mask_max_frames=10,
            freq_mask_max_bins=10,
            num_time_masks=2,
            num_freq_masks=2,
            apply_probability=1.0,
        )
        masked = apply_spec_mix(spec, cfg)
        assert masked.shape == (64, 128)
        assert np.any(masked == 0.0)
        assert np.all(spec == 1.0)  # Original input array is unaffected

    def test_spec_mix_disabled_bypass(self):
        spec = np.ones((64, 128), dtype=np.float32)
        cfg = SpecMixConfig(enabled=False)
        out = apply_spec_mix(spec, cfg)
        assert np.array_equal(out, spec)


class TestCurriculumSchedulerInLoop:
    """Tests curriculum scheduling across training epochs."""

    def test_snr_curriculum_decay_schedule(self):
        cfg = SnrCurriculumConfig(
            enabled=True,
            direction="easy_to_hard",
            start_mean_snr_db=20.0,
            end_mean_snr_db=0.0,
            decay_span_epochs=30,
        )
        assert current_mean_snr(cfg, epoch=0) == pytest.approx(20.0, rel=1e-3)
        assert 0.0 < current_mean_snr(cfg, epoch=15) < 20.0
        assert current_mean_snr(cfg, epoch=30) == pytest.approx(0.0, rel=1e-3)

    def test_snr_curriculum_bounded_sampling(self):
        cfg = SnrCurriculumConfig(
            enabled=True,
            start_mean_snr_db=15.0,
            end_mean_snr_db=5.0,
            std_db=2.0,
        )
        rng = np.random.default_rng(42)
        samples = [sample_snr_for_epoch(cfg, epoch=0, rng=rng) for _ in range(30)]
        for s in samples:
            assert 11.0 <= s <= 19.0


class TestTrainingConfigsAndWeighting:
    """Tests training configurations and dataset sample weighting."""

    def test_all_training_configs_instantiate(self):
        configs = [
            SePrimaryConfig(),
            SeEscalationConfig(),
            SeCrosscheckConfig(),
            ClassifierConfig(),
            AecGateConfig(),
        ]
        for c in configs:
            assert hasattr(c, "model_key")
            assert hasattr(c, "checkpoint_name")
            assert hasattr(c, "mixed_precision")
            assert hasattr(c, "checkpoint_dir")

    def test_weighted_dataset_sampler_construction(self):
        factors = {"tank_tracked": 6.0, "clean_speech": 1.0}
        w_native = compute_sample_weight("tank_tracked", SyncTier.TIER_1_NATIVE_48K, factors)
        assert w_native == 6.0 * 1.00

        w_upsampled = compute_sample_weight("tank_tracked", SyncTier.TIER_3_UPSAMPLED_16K, factors)
        assert w_upsampled == 6.0 * 0.25

"""
Project AEGIS — SOTA Training Upgrades Unit Test Suite
Tests EMA tracking, Worst-Class Checkpoint Selection, SpecMix data augmentation,
and SNR Curriculum Scheduling.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from training.callbacks import (
    EmaConfig,
    EmaTracker,
    WorstClassCheckpointConfig,
    WorstClassCheckpointSelector,
    DISCLOSED_WEAK_CLASSES,
)
from training.data import (
    SpecMixConfig,
    apply_spec_mix,
)
from training.schedulers import (
    SnrCurriculumConfig,
    current_mean_snr,
    sample_snr_for_epoch,
)
from training.configs.base_config import BaseModelConfig
from training.configs.classifier_config import ClassifierConfig
from training.configs.se_primary_config import SePrimaryConfig


class DummyModel:
    """Mock PyTorch module with state_dict for testing EmaTracker."""
    def __init__(self, weights):
        self._weights = weights

    def state_dict(self):
        return {k: v.copy() for k, v in self._weights.items()}


class TestEmaTracker:
    def test_ema_update_math(self):
        w0 = {"w": np.array([1.0, 2.0], dtype=np.float32)}
        model = DummyModel(w0)
        config = EmaConfig(enabled=True, decay=0.9, update_every_n_steps=1)
        ema = EmaTracker(model, config)

        # Update model state
        model._weights["w"] = np.array([2.0, 4.0], dtype=np.float32)
        ema.update(model)

        shadow = ema.state_dict()
        # Expected: 0.9 * [1.0, 2.0] + 0.1 * [2.0, 4.0] = [0.9+0.2, 1.8+0.4] = [1.1, 2.2]
        np.testing.assert_allclose(shadow["w"], np.array([1.1, 2.2], dtype=np.float32), rtol=1e-5)

    def test_ema_disabled(self):
        model = DummyModel({"w": np.array([1.0], dtype=np.float32)})
        config = EmaConfig(enabled=False)
        ema = EmaTracker(model, config)
        assert ema.state_dict() is None


class TestWorstClassCheckpointSelector:
    def test_accepts_when_aggregate_and_classes_improve(self):
        cfg = WorstClassCheckpointConfig(
            enabled=True,
            watched_classes=["rotor_vehicle_drone", "wind"],
            max_allowed_regression_pesq=0.05,
            aggregate_metric_name="pesq_aggregate",
        )
        selector = WorstClassCheckpointSelector(cfg)

        metrics_1 = {
            "pesq_aggregate": 2.50,
            "pesq_rotor_vehicle_drone": 2.40,
            "pesq_wind": 2.30,
        }
        assert selector.should_accept_checkpoint(metrics_1) is True

        metrics_2 = {
            "pesq_aggregate": 2.65,
            "pesq_rotor_vehicle_drone": 2.42,
            "pesq_wind": 2.35,
        }
        assert selector.should_accept_checkpoint(metrics_2) is True

    def test_rejects_when_fragile_class_regresses_beyond_tolerance(self):
        cfg = WorstClassCheckpointConfig(
            enabled=True,
            watched_classes=["rotor_vehicle_drone", "wind"],
            max_allowed_regression_pesq=0.05,
            aggregate_metric_name="pesq_aggregate",
        )
        selector = WorstClassCheckpointSelector(cfg)

        # Step 1: establish baseline
        metrics_1 = {
            "pesq_aggregate": 2.50,
            "pesq_rotor_vehicle_drone": 2.40,
            "pesq_wind": 2.30,
        }
        selector.should_accept_checkpoint(metrics_1)

        # Step 2: Aggregate improves noticeably (+0.20), but wind collapses (-0.15 > 0.05 tolerance)
        metrics_2 = {
            "pesq_aggregate": 2.70,
            "pesq_rotor_vehicle_drone": 2.45,
            "pesq_wind": 2.15,  # Regression of 0.15 from 2.30
        }
        assert selector.should_accept_checkpoint(metrics_2) is False


class TestSpecMixDataAugmentation:
    def test_apply_spec_mix_masking(self):
        spec = np.ones((64, 128), dtype=np.float32)
        cfg = SpecMixConfig(
            enabled=True,
            time_mask_max_frames=10,
            freq_mask_max_bins=10,
            num_time_masks=2,
            num_freq_masks=2,
            apply_probability=1.0,  # force application
        )
        masked = apply_spec_mix(spec, cfg)

        assert masked.shape == (64, 128)
        # Verify that some values were masked to 0.0
        assert np.any(masked == 0.0)
        # Original spectrogram should be untouched
        assert np.all(spec == 1.0)

    def test_spec_mix_disabled(self):
        spec = np.ones((64, 128), dtype=np.float32)
        cfg = SpecMixConfig(enabled=False)
        out = apply_spec_mix(spec, cfg)
        assert np.array_equal(out, spec)


class TestSnrCurriculumScheduler:
    def test_easy_to_hard_decay(self):
        cfg = SnrCurriculumConfig(
            enabled=True,
            direction="easy_to_hard",
            start_mean_snr_db=20.0,
            end_mean_snr_db=0.0,
            decay_span_epochs=30,
        )
        mean_epoch_0 = current_mean_snr(cfg, epoch=0)
        assert mean_epoch_0 == pytest.approx(20.0, rel=1e-3)

        mean_epoch_15 = current_mean_snr(cfg, epoch=15)
        assert 0.0 < mean_epoch_15 < 20.0

        mean_epoch_30 = current_mean_snr(cfg, epoch=30)
        assert mean_epoch_30 == pytest.approx(0.0, rel=1e-3)

    def test_sample_snr_within_bounds(self):
        cfg = SnrCurriculumConfig(
            enabled=True,
            start_mean_snr_db=15.0,
            end_mean_snr_db=5.0,
            std_db=2.0,
        )
        rng = np.random.default_rng(1337)
        samples = [sample_snr_for_epoch(cfg, epoch=0, rng=rng) for _ in range(50)]
        for s in samples:
            # Mean is 15.0, std is 2.0, clipped at +/- 2*std -> [11.0, 19.0]
            assert 11.0 <= s <= 19.0

    def test_disabled_curriculum_raises_error(self):
        cfg = SnrCurriculumConfig(enabled=False)
        with pytest.raises(RuntimeError, match="disabled"):
            current_mean_snr(cfg, 0)


class TestConfigIntegration:
    def test_base_config_contains_all_sota_modules(self):
        cfg = SePrimaryConfig()
        assert hasattr(cfg, "ema")
        assert cfg.ema.enabled is True
        assert hasattr(cfg, "spec_mix")
        assert cfg.spec_mix.enabled is True
        assert hasattr(cfg, "snr_curriculum")
        assert cfg.snr_curriculum.enabled is False  # opt-in by default
        assert hasattr(cfg, "worst_class_checkpoint")
        assert cfg.worst_class_checkpoint.enabled is True

    def test_classifier_disables_ema_by_default(self):
        clf_cfg = ClassifierConfig()
        assert clf_cfg.ema.enabled is False

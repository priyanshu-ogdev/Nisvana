"""
tests/test_model_architectures_and_metrics.py — Tests for training/models and training/utils
"""

import pytest
import torch
from training.models import (
    build_model_for_key,
    DeepFilterNet3Wrapper,
    CleanUMambaWrapper,
    AudioClassifierNet,
    AecFilterNet,
)
from training.utils.metrics import (
    compute_snr_db,
    compute_si_snr_db,
    compute_erle_db,
    compute_pesq_proxy,
    build_eval_metrics_dict,
)
from training.callbacks.worst_class_checkpoint_selector import (
    WorstClassCheckpointConfig,
    WorstClassCheckpointSelector,
)


class TestModelArchitectures:
    """Verifies factory initialization and forward tensor flow for all 5 model architectures."""

    def test_factory_builds_all_models(self):
        m1 = build_model_for_key("aegis-se-primary")
        assert isinstance(m1, DeepFilterNet3Wrapper)
        assert m1.df_lookahead == 0

        m2 = build_model_for_key("aegis-se-escalation")
        assert isinstance(m2, DeepFilterNet3Wrapper)
        assert m2.df_lookahead == 2

        m3 = build_model_for_key("aegis-se-crosscheck")
        assert isinstance(m3, CleanUMambaWrapper)

        m4 = build_model_for_key("aegis-clf-gate")
        assert isinstance(m4, AudioClassifierNet)

        m5 = build_model_for_key("aegis-aec-gate")
        assert isinstance(m5, AecFilterNet)

    def test_factory_rejects_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown model_key"):
            build_model_for_key("nonexistent-model")

    def test_model_forward_shapes(self):
        x = torch.randn(2, 4800)

        m1 = build_model_for_key("aegis-se-primary")
        assert m1(x).shape == (2, 4800)

        m3 = build_model_for_key("aegis-se-crosscheck")
        assert m3(x).shape == (2, 4800)

        m4 = build_model_for_key("aegis-clf-gate")
        assert m4(x).shape == (2, 3)

        m5 = build_model_for_key("aegis-aec-gate")
        farend = torch.randn(2, 4800)
        assert m5(x, farend).shape == (2, 4800)


class TestAudioMetricsAndEvaluator:
    """Verifies numeric evaluation utilities and schema contract."""

    def test_snr_and_si_snr_monotonicity(self):
        clean = torch.randn(2, 8000)
        noisy_high = clean + 0.05 * torch.randn(2, 8000)
        noisy_low = clean + 1.0 * torch.randn(2, 8000)

        snr_high = compute_snr_db(noisy_high, clean)
        snr_low = compute_snr_db(noisy_low, clean)
        assert snr_high > snr_low

        si_high = compute_si_snr_db(noisy_high, clean)
        si_low = compute_si_snr_db(noisy_low, clean)
        assert si_high > si_low

    def test_erle_metric(self):
        mic = torch.randn(2, 4800)
        echo_suppressed = mic * 0.1  # Significant suppression
        erle = compute_erle_db(mic, echo_suppressed)
        assert erle > 15.0  # Approx 20 dB suppression

    def test_pesq_proxy_bounded(self):
        clean = torch.randn(1, 4800)
        assert 1.0 <= compute_pesq_proxy(clean, clean) <= 4.5
        assert 1.0 <= compute_pesq_proxy(torch.randn(1, 4800), clean) <= 4.5

    def test_build_eval_metrics_dict_contract(self):
        est = torch.randn(4, 4800)
        tgt = torch.randn(4, 4800)
        classes = ["wind", "rotor_vehicle_drone", "wind", "tank_tracked"]

        metrics = build_eval_metrics_dict(est, tgt, classes)
        assert "pesq_aggregate" in metrics
        assert "pesq_wind" in metrics
        assert "snr_wind" in metrics
        assert "pesq_rotor_vehicle_drone" in metrics
        assert "snr_rotor_vehicle_drone" in metrics
        assert "pesq_tank_tracked" in metrics
        assert "snr_tank_tracked" in metrics

        # Verify it directly satisfies WorstClassCheckpointSelector without modification
        selector = WorstClassCheckpointSelector(WorstClassCheckpointConfig(
            watched_classes=["wind", "rotor_vehicle_drone"],
        ))
        assert selector.should_accept_checkpoint(metrics) is True

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
    compute_segmental_snr_db,
    compute_stoi,
    compute_pesq,
    compute_pesq_proxy,
    compute_dnsmos_proxy,
    compute_erle_db,
    compute_classifier_metrics,
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

    def test_stoi_monotonicity_and_bounds(self):
        clean = torch.randn(1, 48000)
        noisy_light = clean + 0.05 * torch.randn(1, 48000)
        noisy_heavy = clean + 2.0 * torch.randn(1, 48000)

        stoi_clean = compute_stoi(clean, clean)
        stoi_light = compute_stoi(noisy_light, clean)
        stoi_heavy = compute_stoi(noisy_heavy, clean)

        assert 0.0 <= stoi_clean <= 1.0
        assert 0.0 <= stoi_light <= 1.0
        assert 0.0 <= stoi_heavy <= 1.0
        assert stoi_clean >= stoi_light > stoi_heavy

    def test_segmental_snr_bounds(self):
        clean = torch.randn(1, 48000)
        noisy = clean + 0.1 * torch.randn(1, 48000)
        ssnr = compute_segmental_snr_db(noisy, clean, sr=48000)
        assert -10.0 <= ssnr <= 35.0

    def test_dnsmos_proxy_bounds(self):
        wav = torch.randn(1, 48000)
        mos = compute_dnsmos_proxy(wav, sr=48000)
        assert "dnsmos_sig" in mos
        assert "dnsmos_bak" in mos
        assert "dnsmos_ovrl" in mos
        assert 1.0 <= mos["dnsmos_sig"] <= 5.0
        assert 1.0 <= mos["dnsmos_bak"] <= 5.0
        assert 1.0 <= mos["dnsmos_ovrl"] <= 5.0

    def test_classifier_metrics(self):
        logits = torch.tensor([[5.0, 1.0, 0.0], [0.0, 4.0, 1.0], [1.0, 0.0, 6.0]])
        targets = torch.tensor([0, 1, 2])
        m = compute_classifier_metrics(logits, targets)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["f1_harmonic"] == 1.0

    def test_build_eval_metrics_dict_extended_keys(self):
        est = torch.randn(2, 4800)
        tgt = torch.randn(2, 4800)
        classes = ["tank_tracked", "wind"]
        d = build_eval_metrics_dict(est, tgt, classes)
        assert "stoi_aggregate" in d
        assert "si_snr_aggregate" in d
        assert "stoi_tank_tracked" in d
        assert "si_snr_wind" in d


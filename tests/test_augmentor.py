"""
Project AEGIS — Comprehensive Tests for Grounded Augmentation Engine
"""

import numpy as np
import pytest
from data_forge.config import AugmentationPolicy, UnifiedClass
from data_forge.augmentor import (
    AugmentationEngine,
    AugmentationPolicyEngine,
    BlastOnsetWindow,
    ForbiddenPitchShiftError,
    GainJitter,
    WsolaTimeStretcher,
)


class TestAugmentationPolicies:
    def test_forbidden_pitch_shift_clean_speech(self):
        with pytest.raises(ForbiddenPitchShiftError):
            AugmentationPolicyEngine.assert_pitch_shift_permitted(UnifiedClass.CLEAN_SPEECH)

    def test_forbidden_pitch_shift_tank_tracked(self):
        with pytest.raises(ForbiddenPitchShiftError):
            AugmentationPolicyEngine.assert_pitch_shift_permitted(UnifiedClass.TANK_TRACKED)

    def test_forbidden_pitch_shift_howitzer_artillery(self):
        with pytest.raises(ForbiddenPitchShiftError):
            AugmentationPolicyEngine.assert_pitch_shift_permitted(UnifiedClass.ARTILLERY_HOWITZER)

    def test_forbidden_pitch_shift_jet_cockpit(self):
        with pytest.raises(ForbiddenPitchShiftError):
            AugmentationPolicyEngine.assert_pitch_shift_permitted(UnifiedClass.JET_COCKPIT)

    def test_forbidden_pitch_shift_explosion_blast(self):
        with pytest.raises(ForbiddenPitchShiftError):
            AugmentationPolicyEngine.assert_pitch_shift_permitted(UnifiedClass.EXPLOSION_BLAST)

    def test_forbidden_pitch_shift_gunshot(self):
        with pytest.raises(ForbiddenPitchShiftError):
            AugmentationPolicyEngine.assert_pitch_shift_permitted(UnifiedClass.GUNSHOT_FIREARM)

    def test_time_stretch_rate_validation(self):
        # Rates within [0.90, 1.10] must pass
        assert AugmentationPolicyEngine.validate_time_stretch_rate(1.05) == 1.05
        assert AugmentationPolicyEngine.validate_time_stretch_rate(0.95) == 0.95

        # Extreme rates must be rejected
        with pytest.raises(ValueError):
            AugmentationPolicyEngine.validate_time_stretch_rate(1.35)
        with pytest.raises(ValueError):
            AugmentationPolicyEngine.validate_time_stretch_rate(0.60)

    def test_gain_jitter_validation(self):
        assert AugmentationPolicyEngine.validate_gain_jitter_db(2.0) == 2.0
        with pytest.raises(ValueError):
            AugmentationPolicyEngine.validate_gain_jitter_db(6.0)


class TestWsolaTimeStretch:
    def test_wsola_preserves_length_and_stability(self):
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

        stretcher = WsolaTimeStretcher(sample_rate=sr)
        stretched_fast = stretcher.process(audio, rate=1.06)

        # Output should be shorter (duration / 1.06)
        expected_len = int(len(audio) / 1.06)
        assert abs(len(stretched_fast) - expected_len) < 2000
        assert not np.isnan(stretched_fast).any()


class TestBlastOnsetWindow:
    def test_blast_onset_preservation(self):
        sr = 48000
        # Create a synthetic blast impulse at t=0.2s
        audio = np.zeros(int(1.0 * sr), dtype=np.float32)
        onset_idx = int(0.2 * sr)
        audio[onset_idx : onset_idx + 10] = 0.9  # Sharp rise
        decay = np.exp(-np.linspace(0, 5, 2000))
        audio[onset_idx : onset_idx + 2000] = 0.9 * decay

        windower = BlastOnsetWindow(sample_rate=sr)
        found_onset = windower.find_onset(audio)
        assert abs(found_onset - onset_idx) <= 5

        # Create windowed variation
        windowed = windower.process(audio, target_length_samples=int(0.5 * sr), onset_offset_samples=int(0.05 * sr))
        assert len(windowed) == int(0.5 * sr)
        assert np.max(windowed) > 0.8

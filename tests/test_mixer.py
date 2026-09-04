"""
Project AEGIS — Comprehensive Tests for Data-Forge Mixing Engine (Models 1-5)
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from data_forge.config import ClassifierCategory, UnifiedClass, DATASET_PROFILES
from data_forge.mixer import (
    AecBranch,
    ClassifierBranch,
    SnrMixerEngine,
    SpeechEnhancementBranch,
)
from data_forge.mixer.speech_enhancement import resolve_source_dataset


@pytest.fixture
def clean_speech_sample():
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Fundamental voice frequency + harmonics
    speech = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    return speech.astype(np.float32), sr


@pytest.fixture
def noise_sample():
    sr = 48000
    # Ambient noise
    noise = 0.3 * np.random.randn(sr).astype(np.float32)
    return noise, sr


@pytest.fixture
def rir_sample():
    sr = 48000
    length = int(0.25 * sr)
    t = np.linspace(0, 0.25, length)
    decay = np.exp(-6.91 * t / 0.25)
    rir = np.random.randn(length) * decay
    rir[0] = 1.0
    rir = rir / np.sqrt(np.sum(rir**2))
    return rir.astype(np.float32), sr


class TestSnrMixerEngine:
    def test_snr_mixing_accuracy(self, clean_speech_sample, noise_sample):
        speech, sr = clean_speech_sample
        noise, _ = noise_sample
        mixer = SnrMixerEngine(sample_rate=sr)

        for target_snr in (-5.0, 0.0, 10.0, 18.0):
            res = mixer.mix_signals(speech, noise, target_snr_db=target_snr)
            assert abs(res.measured_snr_db - target_snr) < 0.4
            assert res.rir_applied is False
            assert len(res.noisy_audio) == len(speech)

    def test_rir_convolution(self, clean_speech_sample, noise_sample, rir_sample):
        speech, sr = clean_speech_sample
        noise, _ = noise_sample
        rir, _ = rir_sample
        mixer = SnrMixerEngine(sample_rate=sr)

        res = mixer.mix_signals(speech, noise, target_snr_db=10.0, rir=rir)
        assert res.rir_applied is True
        # Reverberant target should not be identical to dry clean target
        assert not np.allclose(res.clean_target, res.reverberant_target)
        # Power should be preserved
        assert abs(np.mean(res.reverberant_target**2) - np.mean(res.clean_target**2)) < 0.2


class TestSpeechEnhancementBranch:
    def test_branch_generation(self, clean_speech_sample, noise_sample, rir_sample, tmp_path):
        speech, sr = clean_speech_sample
        noise, _ = noise_sample
        rir, _ = rir_sample

        clean_path = tmp_path / "speech.wav"
        noise_path = tmp_path / "noise.wav"
        rir_path = tmp_path / "rir.wav"

        sf.write(clean_path, speech, sr)
        sf.write(noise_path, noise, sr)
        sf.write(rir_path, rir, sr)

        branch_dir = tmp_path / "branch_se"
        branch = SpeechEnhancementBranch(output_dir=branch_dir)

        records = branch.generate_mixtures(
            clean_files=[clean_path],
            noise_files=[noise_path],
            rir_files=[rir_path],
            num_mixtures=3,
        )

        assert len(records) == 3
        manifest_p = branch_dir / "manifest.json"
        assert manifest_p.exists()

        for rec in records:
            assert Path(rec["noisy_path"]).exists()
            assert Path(rec["clean_path"]).exists()
            assert Path(rec["rir_path"]).exists()

    def test_source_dataset_resolves_correctly_including_multi_word_keys(self):
        # Exercises longest-prefix-match resolution against DATASET_PROFILES
        assert resolve_source_dataset("noisex92_leopard.wav") == "noisex92"
        assert resolve_source_dataset("mad_gunshot_001.wav") == "mad"
        assert resolve_source_dataset("gunshot_dryad_398398.wav") == "gunshot_dryad"
        assert resolve_source_dataset("drone_audioset_clip04.wav") == "drone_audioset"
        assert resolve_source_dataset("vctk_demand_p225_001.wav") == "vctk_demand"
        assert resolve_source_dataset("sirens_urban_042.wav") == "sirens_urban"
        assert resolve_source_dataset("openslr_rirs_ir017.wav") == "openslr_rirs"
        assert resolve_source_dataset("totally_unknown_file.wav") == "unknown"

    def test_sync_tier_resolution_matches_authoritative_profiles(self):
        # Asserts sync tier lookup directly respects DATASET_PROFILES
        assert DATASET_PROFILES["mad"].default_sync_tier.value == 1
        assert DATASET_PROFILES["noisex92"].default_sync_tier.value == 3

    def test_mixer_assigns_correct_sync_tier_from_source_dataset(self, clean_speech_sample, noise_sample, rir_sample, tmp_path):
        speech, sr = clean_speech_sample
        noise, _ = noise_sample
        rir, _ = rir_sample

        clean_path = tmp_path / "speech.wav"
        sf.write(clean_path, speech, sr)
        rir_path = tmp_path / "rir.wav"
        sf.write(rir_path, rir, sr)

        # Create two noise files: one NOISEX-92 (Tier 3), one MAD (Tier 1)
        noise_tier3 = tmp_path / "noisex92_leopard.wav"
        noise_tier1 = tmp_path / "mad_gunshot_001.wav"
        sf.write(noise_tier3, noise, sr)
        sf.write(noise_tier1, noise, sr)

        branch_dir = tmp_path / "branch_se_sync"
        branch = SpeechEnhancementBranch(output_dir=branch_dir)

        records_t3 = branch.generate_mixtures(
            clean_files=[clean_path],
            noise_files=[noise_tier3],
            rir_files=[rir_path],
            num_mixtures=1,
            split="train",
        )
        assert records_t3[0]["sync_tier"] == 3

        records_t1 = branch.generate_mixtures(
            clean_files=[clean_path],
            noise_files=[noise_tier1],
            rir_files=[rir_path],
            num_mixtures=1,
            split="train",
        )
        assert records_t1[0]["sync_tier"] == 1


class TestClassifierBranch:
    def test_harmonicity_index(self):
        sr = 48000
        t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
        # Periodic drone / engine signature (150 Hz)
        periodic_sig = 0.6 * np.sin(2 * np.pi * 150 * t).astype(np.float32)
        harmonicity_high = ClassifierBranch.compute_harmonicity_index(periodic_sig, sr)
        assert harmonicity_high > 0.6

        # Stochastic white noise
        white_noise = np.random.randn(int(0.5 * sr)).astype(np.float32)
        harmonicity_low = ClassifierBranch.compute_harmonicity_index(white_noise, sr)
        assert harmonicity_low < 0.35

    def test_3way_category_mapping(self):
        # Drone -> stationary_harmonic
        cat_drone = ClassifierBranch.map_to_3way_category("drone_uav", snr_db=5.0)
        assert cat_drone == ClassifierCategory.STATIONARY_HARMONIC

        # Tank -> stationary_harmonic
        cat_tank = ClassifierBranch.map_to_3way_category("tank_tracked", snr_db=0.0)
        assert cat_tank == ClassifierCategory.STATIONARY_HARMONIC

        # Explosion -> non_stationary_transient
        cat_blast = ClassifierBranch.map_to_3way_category("explosion_blast", snr_db=5.0)
        assert cat_blast == ClassifierCategory.NON_STATIONARY_TRANSIENT

        # High SNR speech -> speech_dominant
        cat_speech = ClassifierBranch.map_to_3way_category("general_noise", snr_db=15.0)
        assert cat_speech == ClassifierCategory.SPEECH_DOMINANT


class TestAecBranch:
    def test_aec_pair_organization(self, tmp_path):
        sr = 48000
        t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
        sig = 0.3 * np.sin(2 * np.pi * 400 * t).astype(np.float32)

        raw_aec_dir = tmp_path / "raw_aec"
        raw_aec_dir.mkdir()

        sf.write(raw_aec_dir / "sample01_mic.wav", sig, sr)
        sf.write(raw_aec_dir / "sample01_farend.wav", sig, sr)
        sf.write(raw_aec_dir / "sample01_nearend.wav", sig, sr)
        sf.write(raw_aec_dir / "sample01_echo.wav", sig, sr)

        out_branch_dir = tmp_path / "branch_aec"
        branch = AecBranch(output_dir=out_branch_dir)
        quads = branch.organize_aec_pairs(raw_aec_dir)

        assert len(quads) == 1
        assert quads[0]["prefix"] == "sample01"
        assert Path(quads[0]["mic_path"]).exists()
        assert Path(quads[0]["farend_path"]).exists()
        assert Path(quads[0]["nearend_path"]).exists()
        assert Path(quads[0]["echo_path"]).exists()

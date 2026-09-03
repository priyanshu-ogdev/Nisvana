"""
Project AEGIS — Comprehensive Tests for 10-Step Preprocessing Pipeline
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from data_forge.config import SyncTier, TARGET_SAMPLE_RATE, UnifiedClass
from data_forge.preprocessor import (
    ChannelStandardizer,
    Deduplicator,
    FormatStandardizer,
    IntegrityChecker,
    LicenseComplianceMode,
    LicenseFilter,
    LoudnessNormalizer,
    MetadataTagger,
    Resampler,
    Splitter,
    VadValidator,
)


@pytest.fixture
def sample_sine_audio():
    """Generates a 1-second 440Hz sine wave at 48kHz."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32), sr


@pytest.fixture
def sample_16k_audio():
    """Generates a 1-second sine wave at 16kHz."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.4 * np.sin(2 * np.pi * 300 * t)
    return audio.astype(np.float32), sr


class TestStep1Format:
    def test_format_standardization(self, sample_sine_audio, tmp_path):
        audio, sr = sample_sine_audio
        test_wav = tmp_path / "test_orig.wav"
        sf.write(test_wav, audio, sr)

        standardizer = FormatStandardizer(target_subtype="PCM_16")
        out_wav = tmp_path / "test_standardized.wav"
        data, out_sr = standardizer.process(test_wav, output_path=out_wav)

        assert out_wav.exists()
        assert out_sr == sr
        info = sf.info(out_wav)
        assert info.subtype == "PCM_16"
        assert info.format == "WAV"


class TestStep2Resample:
    def test_native_48k_pass_through(self, sample_sine_audio):
        audio, sr = sample_sine_audio
        resampler = Resampler(target_sr=48000)
        res_audio, out_sr, tier = resampler.process(audio, orig_sr=48000)

        assert out_sr == 48000
        assert tier == SyncTier.TIER_1_NATIVE_48K
        assert np.allclose(audio, res_audio)

    def test_16k_to_48k_polyphase_upsampling(self, sample_16k_audio):
        audio, orig_sr = sample_16k_audio
        resampler = Resampler(target_sr=48000)
        res_audio, out_sr, tier = resampler.process(audio, orig_sr=orig_sr)

        assert out_sr == 48000
        assert tier == SyncTier.TIER_3_UPSAMPLED_16K
        assert len(res_audio) == 48000

    def test_44k1_resampling(self):
        sr_orig = 44100
        t = np.linspace(0, 1.0, sr_orig, endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)

        resampler = Resampler(target_sr=48000)
        res_audio, out_sr, tier = resampler.process(audio, orig_sr=sr_orig)

        assert out_sr == 48000
        assert tier == SyncTier.TIER_2_RESAMPLED_44K
        assert abs(len(res_audio) - 48000) <= 2


class TestStep3Loudness:
    def test_loudness_normalization(self, sample_sine_audio):
        audio, sr = sample_sine_audio
        normalizer = LoudnessNormalizer(target_lufs=-23.0)
        norm_audio, measured_lufs, true_peak = normalizer.process(audio)

        # After normalization, integrated LUFS should be close to -23 LUFS
        meter = normalizer.meter
        post_lufs = meter.integrated_loudness(norm_audio)
        assert abs(post_lufs - (-23.0)) < 0.8
        assert true_peak <= -1.0  # Respects peak ceiling

    def test_short_transient_fallback(self):
        # 50ms impulse click
        audio = np.zeros(2400, dtype=np.float32)
        audio[100:150] = 0.8
        normalizer = LoudnessNormalizer(target_lufs=-23.0)
        norm_audio, measured_lufs, true_peak = normalizer.process(audio)

        assert len(norm_audio) == len(audio)
        assert not np.isnan(measured_lufs)
        assert true_peak <= -1.0


class TestStep4Vad:
    def test_silence_trimming(self):
        sr = 48000
        # 0.5s silence + 0.8s tone + 0.5s silence
        lead_silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        tone_t = np.linspace(0, 0.8, int(0.8 * sr), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 400 * tone_t).astype(np.float32)
        trail_silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        audio = np.concatenate([lead_silence, tone, trail_silence])

        vad = VadValidator(sample_rate=sr, min_active_sec=0.4)
        is_valid, trimmed, msg = vad.process(audio)

        assert is_valid is True
        # Trimmed length should be significantly shorter than original (1.8s)
        assert len(trimmed) < len(audio)
        assert len(trimmed) >= int(0.8 * sr)

    def test_dead_recording_rejection(self):
        sr = 48000
        silent_audio = np.zeros(sr * 2, dtype=np.float32)
        vad = VadValidator(sample_rate=sr)
        is_valid, _, msg = vad.process(silent_audio)
        assert is_valid is False
        assert "silence threshold" in msg


class TestStep5Channel:
    def test_multichannel_to_mono_downmix(self):
        # Stereo audio
        sr = 48000
        t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
        ch0 = 0.4 * np.sin(2 * np.pi * 440 * t)
        ch1 = 0.4 * np.sin(2 * np.pi * 880 * t)
        stereo = np.stack([ch0, ch1], axis=1).astype(np.float32)

        standardizer = ChannelStandardizer(mode="mono_mean")
        mono, channels = standardizer.process(stereo)

        assert channels == 1
        assert mono.ndim == 1
        assert len(mono) == len(stereo)


class TestStep6Integrity:
    def test_clean_audio(self, sample_sine_audio):
        audio, _ = sample_sine_audio
        checker = IntegrityChecker()
        report = checker.process(audio)
        assert report.is_clean is True
        assert report.has_nan is False
        assert report.has_inf is False

    def test_nan_detection(self):
        audio = np.array([0.1, 0.2, np.nan, 0.4], dtype=np.float32)
        checker = IntegrityChecker()
        report = checker.process(audio)
        assert report.is_clean is False
        assert report.has_nan is True

    def test_severe_clipping_detection(self):
        audio = np.ones(1000, dtype=np.float32)  # 100% clipped at 1.0
        checker = IntegrityChecker(max_allowed_clip_ratio=0.01)
        report = checker.process(audio)
        assert report.is_clean is False
        assert "hard-clipping" in report.rejection_reason


class TestStep7Metadata:
    def test_metadata_tagging(self):
        tagger = MetadataTagger()
        meta = tagger.create_metadata(
            clip_id="noisex_leopard_001",
            filename="noisex_leopard_001.wav",
            source_dataset="noisex92",
            unified_class=UnifiedClass.TANK_TRACKED,
            sync_tier=SyncTier.TIER_3_UPSAMPLED_16K,
            duration_sec=5.2,
            sample_rate=48000,
            channels=1,
            measured_lufs=-23.0,
            true_peak_dbfs=-1.0,
        )
        assert meta.source_dataset == "noisex92"
        assert meta.unified_class == "tank_tracked"
        assert meta.sync_tier == 3
        assert meta.augmentation_policy == "moderate_no_pitch"


class TestStep8Dedup:
    def test_identical_audio_deduplication(self, sample_sine_audio):
        audio, sr = sample_sine_audio
        dedup = Deduplicator()
        fp1 = dedup.compute_fingerprint(audio, sr)
        fp2 = dedup.compute_fingerprint(audio.copy(), sr)

        assert fp1 == fp2
        is_dup1, orig1 = dedup.register_and_check("clip_001", fp1)
        assert is_dup1 is False

        is_dup2, orig2 = dedup.register_and_check("clip_002", fp2)
        assert is_dup2 is True
        assert orig2 == "clip_001"


class TestStep9License:
    def test_commercial_strict_filter(self):
        tagger = MetadataTagger()
        meta_nc = tagger.create_metadata(
            clip_id="esc50_siren",
            filename="esc50_siren.wav",
            source_dataset="sirens_urban",  # CC-BY-NC
            unified_class=UnifiedClass.SIREN_EMERGENCY,
            sync_tier=SyncTier.TIER_2_RESAMPLED_44K,
            duration_sec=3.0,
            sample_rate=48000,
            channels=1,
            measured_lufs=-23.0,
            true_peak_dbfs=-1.0,
        )

        filter_strict = LicenseFilter(mode=LicenseComplianceMode.COMMERCIAL_STRICT)
        keep, reason = filter_strict.filter_clip(meta_nc)
        assert keep is False
        assert "Non-Commercial" in reason

        filter_research = LicenseFilter(mode=LicenseComplianceMode.RESEARCH_PROTOTYPE)
        keep_res, _ = filter_research.filter_clip(meta_nc)
        assert keep_res is True


class TestStep10Split:
    def test_generalization_isolation(self):
        tagger = MetadataTagger()
        meta_musan = tagger.create_metadata(
            clip_id="musan_noise_01",
            filename="musan_noise_01.wav",
            source_dataset="musan_generalization",
            unified_class=UnifiedClass.GENERAL_NOISE,
            sync_tier=SyncTier.TIER_3_UPSAMPLED_16K,
            duration_sec=4.0,
            sample_rate=48000,
            channels=1,
            measured_lufs=-23.0,
            true_peak_dbfs=-1.0,
        )

        meta_train = tagger.create_metadata(
            clip_id="dns_clean_01",
            filename="dns_clean_01.wav",
            source_dataset="dns_challenge",
            unified_class=UnifiedClass.CLEAN_SPEECH,
            sync_tier=SyncTier.TIER_1_NATIVE_48K,
            duration_sec=4.0,
            sample_rate=48000,
            channels=1,
            measured_lufs=-23.0,
            true_peak_dbfs=-1.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            splitter = Splitter(splits_dir=Path(tmp_dir))
            splits = splitter.split_clips([meta_musan, meta_train])

            # MUSAN must strictly be in test_generalization
            assert meta_musan in splits["test_generalization"]
            assert meta_musan not in splits["train"]
            assert meta_musan not in splits["val"]


class TestDatasetProfiles:
    def test_mad_sync_tier_is_native_48k(self):
        from data_forge.config import DATASET_PROFILES
        profile = DATASET_PROFILES["mad"]
        assert profile.native_sample_rate == 48000
        assert profile.default_sync_tier == SyncTier.TIER_1_NATIVE_48K

    def test_env_threshold_configuration(self):
        from data_forge.config import (
            TARGET_SAMPLE_RATE,
            TARGET_LUFS,
            TARGET_TRUE_PEAK_DBFS,
            SAMPLES_PER_SHARD,
            SILENCE_ENERGY_THRESHOLD_DB,
        )
        assert TARGET_SAMPLE_RATE == 48000
        assert TARGET_LUFS == -23.0
        assert TARGET_TRUE_PEAK_DBFS == -1.0
        assert SAMPLES_PER_SHARD == 2048
        assert SILENCE_ENERGY_THRESHOLD_DB == -50.0

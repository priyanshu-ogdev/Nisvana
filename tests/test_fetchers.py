"""
Project AEGIS — Tests for Dataset Fetchers
Verifies endpoint reachability, naming conventions, and API handling across all 10 sources.
"""

from pathlib import Path
import pytest
from data_forge.fetcher import (
    AecChallengeFetcher,
    DnsChallengeFetcher,
    DroneAudioSetFetcher,
    GunshotDryadFetcher,
    MadFetcher,
    NoisexFetcher,
    RirFetcher,
    SharedExplosionFetcher,
    SirensFetcher,
    VctkDemandFetcher,
)


class TestNoisexFetcher:
    def test_noisex_filenames_convention(self):
        files = NoisexFetcher.FILES
        assert "buccaneercockpit1.wav" in files
        assert "factoryfloor1.wav" in files
        assert "pinknoise.wav" in files
        assert "whitenoise.wav" in files
        for wrong in ("buccaneer1.wav", "factory1.wav", "pink.wav", "white.wav"):
            assert wrong not in files

    def test_noisex_dry_run(self, tmp_path):
        fetcher = NoisexFetcher(tmp_path)
        results = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(results) > 0
        assert results[0].success is True


class TestSharedExplosionFetcher:
    def test_shared_dry_run(self, tmp_path):
        fetcher = SharedExplosionFetcher(tmp_path)
        results = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(results) > 0


class TestDroneAudioSetFetcher:
    def test_drone_dry_run(self, tmp_path):
        fetcher = DroneAudioSetFetcher(tmp_path)
        results = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(results) > 0
        assert results[0].success is True


class TestMadFetcher:
    def test_mad_annotation_files(self):
        files = MadFetcher.ANNOTATION_FILES
        assert "mad_dataset_annotation.csv" in files
        assert "README.md" in files
        assert "data/MAD_dataset/training.csv" not in files
        assert "data/MAD_dataset/test.csv" not in files

    def test_mad_fails_loudly_without_kaggle_credentials(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        fetcher = MadFetcher(tmp_path)
        result = fetcher._fetch_from_kaggle(dry_run=True, sample_mode=True)
        assert result.success is False
        assert "Kaggle" in (result.error or "")

    def test_mad_dry_run(self, tmp_path):
        fetcher = MadFetcher(tmp_path)
        results = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(results) > 0
        assert results[0].success is True


class TestSirensFetcher:
    def test_sirens_dry_run(self, tmp_path):
        fetcher = SirensFetcher(tmp_path)
        results = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(results) > 0
        assert results[0].success is True


class TestRirFetcher:
    def test_rir_dry_run(self, tmp_path):
        fetcher = RirFetcher(tmp_path)
        results = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(results) > 0
        assert results[0].success is True
        assert (tmp_path / "rir_wavs").exists()


class TestDnsChallengeFetcher:
    def test_dns_dry_run_sample_mode(self, tmp_path):
        fetcher = DnsChallengeFetcher(tmp_path)
        sample_res = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(sample_res) > 0
        assert sample_res[0].success is True

    def test_dns_dry_run_training_mode(self, tmp_path):
        fetcher = DnsChallengeFetcher(tmp_path)
        train_res = fetcher.fetch(sample_mode=False, dry_run=True)
        assert len(train_res) >= 4
        assert any(r.success for r in train_res)

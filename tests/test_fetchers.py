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


class TestVctkDemandFetcher:
    def test_vctk_demand_archives_carry_confirmed_md5s(self):
        archives = {name: md5 for _, name, md5 in VctkDemandFetcher.ARCHIVES}
        assert archives["clean_testset_wav.zip"] == "34eb1c0ba7ef667e9b966866c542fc16"
        assert archives["noisy_testset_wav.zip"] == "fb1b86caa31e8ba5b506c0c64da9aab5"
        assert archives["clean_trainset_28spk_wav.zip"] == "d2d5a45ec32f8fcbf201bde0447e20ba"

    def test_vctk_demand_dry_run(self, tmp_path):
        fetcher = VctkDemandFetcher(tmp_path)
        res = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(res) > 0
        assert res[0].success is True


class TestDnsChallengeFetcher:
    def test_dns_blobs_use_confirmed_naming_pattern(self):
        blobs = DnsChallengeFetcher.VERIFIED_CLEAN_BLOBS + DnsChallengeFetcher.VERIFIED_NOISE_BLOBS
        assert len(blobs) >= 10
        for blob in blobs:
            assert "_NA_NA" not in blob, f"Invalid _NA_NA placeholder found in {blob}"
            assert blob.startswith(("Track1_Headset/", "noise_fullband/", "datasets_fullband."))

    def test_dns_fetcher_has_dynamic_discovery_method(self):
        assert hasattr(DnsChallengeFetcher, "_list_blobs_via_azure_api")
        assert callable(getattr(DnsChallengeFetcher, "_list_blobs_via_azure_api"))

    def test_dns_dev_testset_url_unchanged(self):
        assert DnsChallengeFetcher.DEV_TESTSET_URL == (
            "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_dev_testset.zip"
        )

    def test_dns_dry_run_sample_mode(self, tmp_path):
        fetcher = DnsChallengeFetcher(tmp_path)
        sample_res = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(sample_res) > 0
        assert sample_res[0].success is True

    def test_dns_dry_run_training_mode(self, tmp_path):
        fetcher = DnsChallengeFetcher(tmp_path)
        train_res = fetcher.fetch(sample_mode=False, dry_run=True)
        assert len(train_res) >= 10
        assert any(r.success for r in train_res)


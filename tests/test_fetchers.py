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
        assert "buccaneer1.wav" in files
        assert "factory1.wav" in files
        assert "pink.wav" in files
        assert "white.wav" in files
        assert "leopard.wav" in files
        assert "m109.wav" in files

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

    def test_shared_dataframe_unpacking(self, tmp_path):
        import pickle
        import pandas as pd
        import numpy as np

        pkl_path = tmp_path / "SHAReD.pkl"
        df = pd.DataFrame({
            "audio": [np.random.randn(4800).astype(np.float32) for _ in range(5)],
            "sr": [48000] * 5,
        })
        with open(pkl_path, "wb") as f:
            pickle.dump(df, f)

        fetcher = SharedExplosionFetcher(tmp_path)
        results = fetcher.fetch(sample_mode=False, dry_run=False)
        wav_files = list((tmp_path / "wavs").glob("*.wav"))
        assert len(wav_files) == 5
        assert len(results) == 5
        assert results[0].success is True

    def test_shared_dict_mapping_unpacking(self, tmp_path):
        import pickle
        import numpy as np

        pkl_path = tmp_path / "SHAReD.pkl"
        data = {
            "blast_a": np.random.randn(2400).astype(np.float32),
            "blast_b": {"audio": np.random.randn(2400).astype(np.float32), "sr": 48000},
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        fetcher = SharedExplosionFetcher(tmp_path)
        results = fetcher.fetch(sample_mode=False, dry_run=False)
        wav_files = list((tmp_path / "wavs").glob("*.wav"))
        assert len(wav_files) == 2
        assert len(results) == 2

    def test_shared_ndarray_unpacking(self, tmp_path):
        import pickle
        import numpy as np

        pkl_path = tmp_path / "SHAReD.pkl"
        data = np.random.randn(3, 3000).astype(np.float32)
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        fetcher = SharedExplosionFetcher(tmp_path)
        results = fetcher.fetch(sample_mode=False, dry_run=False)
        wav_files = list((tmp_path / "wavs").glob("*.wav"))
        assert len(wav_files) == 3
        assert len(results) == 3


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
        monkeypatch.delenv("KAGGLE_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")
        fetcher = MadFetcher(tmp_path)
        result = fetcher._fetch_from_kaggle(dry_run=True, sample_mode=True)
        assert result.success is False
        assert "Kaggle" in (result.error or "")

    def test_mad_credentials_detection_with_access_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.setenv("KAGGLE_ACCESS_TOKEN", "KGAT_test_dummy_token")
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")
        fetcher = MadFetcher(tmp_path)
        assert fetcher._kaggle_credentials_present() is True

    def test_mad_credentials_detection_with_legacy_keys(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KAGGLE_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
        monkeypatch.setenv("KAGGLE_KEY", "test_key")
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")
        fetcher = MadFetcher(tmp_path)
        assert fetcher._kaggle_credentials_present() is True

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


class TestAecChallengeFetcher:
    def test_aec_dry_run(self, tmp_path):
        fetcher = AecChallengeFetcher(tmp_path)
        res = fetcher.fetch(sample_mode=True, dry_run=True)
        assert len(res) > 0
        assert res[0].success is True

    def test_aec_quadruplet_structure(self, tmp_path):
        fetcher = AecChallengeFetcher(tmp_path)
        assert len(fetcher.SAMPLE_FILE_IDS) == 5


class TestBaseFetcherResilience:
    def test_domain_token_injection(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_FORGE_GITHUB_TOKEN", "gh_test_token_12345")
        monkeypatch.setenv("DATA_FORGE_HF_TOKEN", "hf_test_token_67890")
        monkeypatch.setenv("DATA_FORGE_DRYAD_API_TOKEN", "dryad_bearer_abcde")
        monkeypatch.setenv("DATA_FORGE_DATAVERSE_API_TOKEN", "dataverse_key_xyz")

        fetcher = NoisexFetcher(tmp_path)
        gh_headers = fetcher.get_headers_for_url("https://raw.githubusercontent.com/speechdnn/Noises/master/test.wav")
        assert gh_headers.get("Authorization") == "token gh_test_token_12345"

        hf_headers = fetcher.get_headers_for_url("https://huggingface.co/datasets/test.parquet")
        assert hf_headers.get("Authorization") == "Bearer hf_test_token_67890"

        dryad_headers = fetcher.get_headers_for_url("https://datadryad.org/api/v2/files/123/download")
        assert dryad_headers.get("Authorization") == "Bearer dryad_bearer_abcde"

        dv_headers = fetcher.get_headers_for_url("https://dataverse.harvard.edu/api/access/datafile/456")
        assert dv_headers.get("X-Dataverse-key") == "dataverse_key_xyz"

    def test_fallback_mirror_failover(self, tmp_path):
        fetcher = NoisexFetcher(tmp_path)
        # Primary is a non-existent endpoint, fallback is an authentic reachable endpoint
        failing_url = "https://raw.githubusercontent.com/speechdnn/Noises/master/NoiseX-92/non_existent_fake_audio_file.wav"
        valid_fallback = "https://raw.githubusercontent.com/speechdnn/Noises/master/NoiseX-92/leopard.wav"

        res = fetcher.download_file(
            failing_url,
            tmp_path / "leopard_test.wav",
            dry_run=True,
            fallback_urls=[valid_fallback],
        )
        assert res.success is True
        assert res.destination == tmp_path / "leopard_test.wav"


class TestFetchManagerOrchestration:
    def test_fetch_manager_has_all_10_sources(self, tmp_path):
        from data_forge.fetcher.manager import FetchManager
        mgr = FetchManager(tmp_path)
        expected_sources = {
            "noisex92", "shared", "gunshot_dryad", "drone_audioset", "mad",
            "vctk_demand", "dns_challenge", "aec_challenge", "sirens_urban", "openslr_rirs"
        }
        assert set(mgr.fetchers.keys()) == expected_sources

    def test_fetch_manager_dry_run_orchestration(self, tmp_path):
        from data_forge.fetcher.manager import FetchManager
        mgr = FetchManager(tmp_path)
        # Test dry-run on a fast subset
        subset = ["noisex92", "drone_audioset", "sirens_urban"]
        for s in subset:
            res = mgr.fetch_source(s, sample_mode=True, dry_run=True)
            assert len(res) > 0
            assert res[0].success is True



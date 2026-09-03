"""
Regression tests for the 2026-09-03 verification/upgrade pass:
- NOISEX-92 filename corrections
- MAD sample-rate/sync-tier correction
- MAD fetcher's Kaggle fallback path exists and fails loudly without creds
- New WebDataset shard exporter round-trips correctly
"""

import tarfile
import tempfile
from pathlib import Path

import pytest

from data_forge.fetcher.noisex import NoisexFetcher
from data_forge.config import DATASET_PROFILES, SyncTier
from data_forge.fetcher.mad import MadFetcher
from data_forge.exporter.shard_writer import ShardWriter


def test_noisex_filenames_match_verified_repo_convention():
    # These four were confirmed wrong against the actual speechdnn/Noises tree
    # (cross-checked against Build-SE-Dataset, UNetGAN-Demo, Extremely-Low-SNR-Demo).
    files = NoisexFetcher.FILES
    assert "buccaneercockpit1.wav" in files
    assert "factoryfloor1.wav" in files
    assert "pinknoise.wav" in files
    assert "whitenoise.wav" in files
    # And the old, wrong names must be gone.
    for wrong in ("buccaneer1.wav", "factory1.wav", "pink.wav", "white.wav"):
        assert wrong not in files


def test_mad_sync_tier_is_native_48k():
    profile = DATASET_PROFILES["mad"]
    assert profile.native_sample_rate == 48000
    assert profile.default_sync_tier == SyncTier.TIER_1_NATIVE_48K


def test_mad_fetcher_no_longer_requests_nonexistent_github_csvs():
    files = MadFetcher.ANNOTATION_FILES
    assert "data/MAD_dataset/training.csv" not in files
    assert "data/MAD_dataset/test.csv" not in files
    assert "mad_dataset_annotation.csv" in files


def test_mad_fetcher_fails_loudly_without_kaggle_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    fetcher = MadFetcher(tmp_path)
    result = fetcher._fetch_from_kaggle(dry_run=True, sample_mode=True)
    assert result.success is False
    assert "Kaggle" in (result.error or "")


def test_shard_writer_round_trip():
    with tempfile.TemporaryDirectory() as d:
        writer = ShardWriter(Path(d), "unit", samples_per_shard=3)
        for i in range(7):
            writer.add_sample(f"{i:06d}", {"wav": b"X" * 10, "json": b"{}"})
        summary = writer.close()

        assert summary["total_samples"] == 7
        assert summary["total_shards"] == 3  # ceil(7/3)

        # Verify the tar contents are actually readable back out.
        first_shard = Path(d) / summary["shards"][0]["shard_file"]
        with tarfile.open(first_shard) as t:
            names = t.getnames()
            assert "000000.wav" in names
            assert "000000.json" in names


def test_pack_branch_to_shards_with_subdirectories():
    from data_forge.exporter.shard_writer import pack_branch_to_shards
    import json

    with tempfile.TemporaryDirectory() as d:
        branch = Path(d) / "se"
        (branch / "noisy").mkdir(parents=True)
        (branch / "clean").mkdir(parents=True)
        (branch / "rir").mkdir(parents=True)
        manifest_data = {
            "samples": [
                {"clip_id": "se_00001"},
                {"clip_id": "se_00002"},
            ]
        }
        (branch / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
        (branch / "noisy" / "se_00001_noisy.wav").write_bytes(b"noisy_1")
        (branch / "clean" / "se_00001_clean.wav").write_bytes(b"clean_1")
        (branch / "noisy" / "se_00002_noisy.wav").write_bytes(b"noisy_2")
        (branch / "clean" / "se_00002_clean.wav").write_bytes(b"clean_2")

        def se_groups(sample_root: Path):
            k = sample_root.name
            b = sample_root.parent
            n = b / "noisy" / f"{k}_noisy.wav"
            c = b / "clean" / f"{k}_clean.wav"
            if n.exists() and c.exists():
                return {"noisy.wav": n, "clean.wav": c}
            return None

        out_shards = Path(d) / "shards"
        summary = pack_branch_to_shards(branch, out_shards, "se", se_groups, samples_per_shard=2)
        assert summary["samples_packed"] == 2
        assert summary["total_shards"] == 1


def test_dataset_card_generation(tmp_path):
    from data_forge.exporter import write_dataset_card, generate_dataset_card

    card = generate_dataset_card({"speech_enhancement": 100})
    assert "pretty_name: Project AEGIS Defence ANC Training Corpus" in card
    assert "NOISEX-92" in card
    assert "Military Audio Dataset (MAD)" in card

    out_file = tmp_path / "CARD.md"
    write_dataset_card(out_file, {"speech_enhancement": 100})
    assert out_file.exists()
    assert len(out_file.read_text(encoding="utf-8")) > 100

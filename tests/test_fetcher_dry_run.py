"""
Project AEGIS — Tests for Fetcher Endpoints (Dry-Run Mode)
Verifies that dataset URLs are responsive without downloading multi-gigabyte files.
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


def test_noisex_dry_run(tmp_path):
    fetcher = NoisexFetcher(tmp_path)
    results = fetcher.fetch(sample_mode=True, dry_run=True)
    assert len(results) > 0
    assert results[0].success is True


def test_shared_dry_run(tmp_path):
    fetcher = SharedExplosionFetcher(tmp_path)
    results = fetcher.fetch(sample_mode=True, dry_run=True)
    assert len(results) > 0


def test_drone_dry_run(tmp_path):
    fetcher = DroneAudioSetFetcher(tmp_path)
    results = fetcher.fetch(sample_mode=True, dry_run=True)
    assert len(results) > 0
    assert results[0].success is True


def test_mad_dry_run(tmp_path):
    fetcher = MadFetcher(tmp_path)
    results = fetcher.fetch(sample_mode=True, dry_run=True)
    assert len(results) > 0
    assert results[0].success is True


def test_sirens_dry_run(tmp_path):
    fetcher = SirensFetcher(tmp_path)
    results = fetcher.fetch(sample_mode=True, dry_run=True)
    assert len(results) > 0
    assert results[0].success is True


def test_rir_dry_run(tmp_path):
    fetcher = RirFetcher(tmp_path)
    results = fetcher.fetch(sample_mode=True, dry_run=True)
    assert len(results) > 0
    assert results[0].success is True
    assert (tmp_path / "rir_wavs").exists()


def test_dns_dry_run(tmp_path):
    fetcher = DnsChallengeFetcher(tmp_path)
    # Test sample mode
    sample_res = fetcher.fetch(sample_mode=True, dry_run=True)
    assert len(sample_res) > 0
    assert sample_res[0].success is True

    # Test full training mode dry-run
    train_res = fetcher.fetch(sample_mode=False, dry_run=True)
    assert len(train_res) >= 4
    assert any(r.success for r in train_res)

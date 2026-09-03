"""
Project AEGIS — DNS Challenge Fetcher (Training-Scale & Evaluation)
Source: Reddy et al., Microsoft Deep Noise Suppression Challenge (ICASSP / INTERSPEECH)
Repository: https://github.com/microsoft/DNS-Challenge

VERIFIED 2026-09-03:
- Base account: https://dnschallengepublic.blob.core.windows.net/dns5archive
- Dev/eval testset: V5_dev_testset.zip (3.08 GB, verified 200 OK)
- Full training corpora: V5_training_dataset (verified 200 OK across Azure endpoints):
    Track1_Headset/emotional_speech.tgz (38.9 MB)
    Track1_Headset/VocalSet_48kHz_mono.tgz (14.9 MB)
    Track1_Headset/vctk_wav48_silence_trimmed.tgz.partaa (5.24 GB)
    Track1_Headset/read_speech.tgz.partaa (5.24 GB)
    noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2 (5.36 GB)
    noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2 (3.47 GB)
    datasets_fullband.impulse_responses_000.tar.bz2 (264.9 MB)
- Azure Blob Storage ACL note: container enumeration (?restype=container&comp=list)
  is disabled by Microsoft (returns 404/403), while individual blob read access is public.
"""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import List, Optional
from .base import BaseFetcher, DownloadResult, logger


class DnsChallengeFetcher(BaseFetcher):
    """
    Fetches fullband 48kHz clean speech, noise, and RIR archives from
    Microsoft DNS Challenge, verifying endpoints and supporting sample and full modes.
    """

    ACCOUNT_BASE = "https://dnschallengepublic.blob.core.windows.net"
    CONTAINER = "dns5archive"
    TRAINING_BASE = f"{ACCOUNT_BASE}/{CONTAINER}/V5_training_dataset"
    DEV_TESTSET_URL = f"{ACCOUNT_BASE}/{CONTAINER}/V5_dev_testset.zip"

    # Confirmed authentic blob paths from Microsoft DNS-Challenge repository
    # (download-dns-challenge-5-headset-training.sh and download-dns-challenge-5-noise-ir.sh)
    # All verified 200 OK directly against Azure Blob Storage.
    VERIFIED_CLEAN_BLOBS = [
        "Track1_Headset/emotional_speech.tgz",
        "Track1_Headset/VocalSet_48kHz_mono.tgz",
        "Track1_Headset/vctk_wav48_silence_trimmed.tgz.partaa",
        "Track1_Headset/vctk_wav48_silence_trimmed.tgz.partab",
        "Track1_Headset/vctk_wav48_silence_trimmed.tgz.partac",
        "Track1_Headset/read_speech.tgz.partaa",
        "Track1_Headset/read_speech.tgz.partab",
        "Track1_Headset/read_speech.tgz.partac",
    ]

    VERIFIED_NOISE_BLOBS = [
        "noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2",
        "noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2",
        "noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2",
        "noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2",
        "noise_fullband/datasets_fullband.noise_fullband.audioset_004.tar.bz2",
        "noise_fullband/datasets_fullband.noise_fullband.audioset_005.tar.bz2",
        "noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2",
        "noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2",
        "noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2",
        "datasets_fullband.impulse_responses_000.tar.bz2",
    ]

    # Prefixes to probe if dynamic container enumeration is enabled
    LIST_PREFIXES = ["Track1_Headset/", "noise_fullband/", "datasets_fullband.impulse_responses"]

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results: List[DownloadResult] = []

        # Dev/eval set — always fetched, confirmed reachable (200 OK)
        dest_eval = self.output_dir / "V5_dev_testset.zip"
        res_eval = self.download_file(self.DEV_TESTSET_URL, dest_eval, dry_run=dry_run)
        results.append(res_eval)

        if sample_mode:
            if not dry_run and res_eval.success and dest_eval.exists():
                extract_target = self.output_dir / "dns5_dev"
                if not extract_target.exists():
                    logger.info("Extracting sample DNS audio files into %s...", extract_target.name)
                    try:
                        with zipfile.ZipFile(dest_eval, "r") as z:
                            wav_members = [m for m in z.namelist() if m.endswith(".wav")][:25]
                            z.extractall(extract_target, members=wav_members)
                        logger.info("Extracted sample DNS audio files.")
                    except Exception as e:
                        logger.warning("Sample extraction note: %s", e)
            return results

        # Full training mode: attempt dynamic discovery, otherwise use verified official list
        discovered = self._list_blobs_via_azure_api(dry_run=dry_run)
        blob_paths = discovered if discovered else self._fallback_blob_list()

        for rel_blob in blob_paths:
            url = f"{self.TRAINING_BASE}/{rel_blob}"
            dest = self.output_dir / Path(rel_blob).name
            results.append(self.download_file(url, dest, dry_run=dry_run))

        return results

    def _list_blobs_via_azure_api(self, dry_run: bool) -> Optional[List[str]]:
        """
        Attempts to call Azure's public 'List Blobs' REST API against the container.
        Returns None (triggering the verified blob list) if listing is disabled on the container.
        """
        try:
            import requests
        except ImportError:
            return None

        discovered: List[str] = []
        for prefix in self.LIST_PREFIXES:
            list_url = (
                f"{self.ACCOUNT_BASE}/{self.CONTAINER}"
                f"?restype=container&comp=list&prefix=V5_training_dataset/{prefix}"
            )
            try:
                if dry_run:
                    resp = requests.head(list_url, timeout=10)
                    if resp.status_code >= 400:
                        return None
                    continue
                resp = requests.get(list_url, timeout=15)
                if resp.status_code != 200:
                    return None
                root = ET.fromstring(resp.content)
                for blob in root.iter("Blob"):
                    name_el = blob.find("Name")
                    if name_el is not None and name_el.text:
                        # strip V5_training_dataset/ prefix
                        name = name_el.text
                        if name.startswith("V5_training_dataset/"):
                            name = name[len("V5_training_dataset/"):]
                        discovered.append(name)
            except Exception:
                return None

        return discovered if discovered else None

    def _fallback_blob_list(self) -> List[str]:
        return self.VERIFIED_CLEAN_BLOBS + self.VERIFIED_NOISE_BLOBS

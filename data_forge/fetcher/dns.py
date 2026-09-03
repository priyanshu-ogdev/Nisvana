"""
Project AEGIS — DNS Challenge Fetcher (Training-Scale & Evaluation)
Source: Reddy et al., Microsoft Deep Noise Suppression Challenge (ICASSP / INTERSPEECH)
Repository: https://github.com/microsoft/DNS-Challenge
"""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from .base import BaseFetcher, DownloadResult, logger


class DnsChallengeFetcher(BaseFetcher):
    """
    Fetches fullband 48kHz clean speech, noise, and RIR archives from Microsoft DNS Challenge.
    Supports:
    - Sample / Eval Mode: Downloads V5_dev_testset.zip (~3.08 GB)
    - Full Training / Server Mode: Downloads multi-gigabyte/multi-part archives from Azure Blob Storage:
      * Complete 39 GB noise corpus (AudioSet 000-006 + Freesound 000-001)
      * Room impulse responses (datasets_fullband.impulse_responses_000)
      * Native 48kHz clean speech (VCTK wav48, VocalSet 48k, Emotional speech, Read speech)
    """

    AZURE_TRAINING_BASE = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
    DEV_TESTSET_URL = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_dev_testset.zip"

    # Verified 2026-09-03: Exact blob names from microsoft/DNS-Challenge official download scripts
    NOISE_BLOBS = [
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

    CLEAN_SPEECH_BLOBS = [
        "Track1_Headset/VocalSet_48kHz_mono.tgz",
        "Track1_Headset/emotional_speech.tgz",
        "Track1_Headset/vctk_wav48_silence_trimmed.tgz.partaa",
        "Track1_Headset/vctk_wav48_silence_trimmed.tgz.partab",
        "Track1_Headset/vctk_wav48_silence_trimmed.tgz.partac",
        # Primary English read speech shards
        "Track1_Headset/read_speech.tgz.partaa",
        "Track1_Headset/read_speech.tgz.partab",
        "Track1_Headset/read_speech.tgz.partac",
    ]

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results: List[DownloadResult] = []

        if sample_mode:
            logger.info("DNS Challenge: running in sample/eval mode (fetching V5 dev testset)...")
            dest = self.output_dir / "V5_dev_testset.zip"
            res = self.download_file(self.DEV_TESTSET_URL, dest, dry_run=dry_run)
            results.append(res)

            if not dry_run and res.success and dest.exists():
                extract_target = self.output_dir / "dns5_dev"
                if not extract_target.exists():
                    logger.info("Extracting DNS-5 testset audio...")
                    with zipfile.ZipFile(dest, "r") as z:
                        wav_files = [m for m in z.namelist() if m.endswith(".wav")][:25]
                        z.extractall(extract_target, members=wav_files)
                    logger.info("Extracted sample DNS audio files into %s", extract_target)

            return results

        # Full training mode / server mode
        logger.info("DNS Challenge: running in FULL TRAINING mode on Azure Blob archives...")

        # 1. Probe or download dev testset
        dest_eval = self.output_dir / "V5_dev_testset.zip"
        res_eval = self.download_file(self.DEV_TESTSET_URL, dest_eval, dry_run=dry_run)
        results.append(res_eval)

        # 2. Download Noise Archives (~39 GB)
        noise_dir = self.output_dir / "noise_fullband"
        noise_dir.mkdir(parents=True, exist_ok=True)

        blobs_to_fetch = self.NOISE_BLOBS + self.CLEAN_SPEECH_BLOBS
        if dry_run:
            # In dry-run mode, probe sample of noise and clean blobs to verify reachability
            blobs_to_probe = [
                self.NOISE_BLOBS[0],
                self.NOISE_BLOBS[-1],
                self.CLEAN_SPEECH_BLOBS[0],
                self.CLEAN_SPEECH_BLOBS[2],
            ]
            for rel_blob in blobs_to_probe:
                url = f"{self.AZURE_TRAINING_BASE}/{rel_blob}"
                filename = Path(rel_blob).name
                dest = self.output_dir / filename
                r = self.download_file(url, dest, dry_run=True)
                results.append(r)
            return results

        # Live download of noise and clean speech blobs
        multi_parts: Dict[str, List[Path]] = {}

        for rel_blob in blobs_to_fetch:
            url = f"{self.AZURE_TRAINING_BASE}/{rel_blob}"
            filename = Path(rel_blob).name
            dest = self.output_dir / filename

            res = self.download_file(url, dest, dry_run=False)
            results.append(res)

            if res.success and dest.exists():
                # Check for multi-part archives (.partaa, .partab)
                if ".part" in filename:
                    base_archive = filename.split(".part")[0]
                    if base_archive not in multi_parts:
                        multi_parts[base_archive] = []
                    multi_parts[base_archive].append(dest)
                elif filename.endswith((".tar.bz2", ".tgz", ".tar.gz")):
                    self._extract_archive(dest, self.output_dir / "extracted")

        # Concatenate and extract multi-part archives
        for base_name, parts in multi_parts.items():
            parts.sort()
            combined_file = self.output_dir / base_name
            if not combined_file.exists():
                logger.info("Concatenating %d parts for %s...", len(parts), base_name)
                with open(combined_file, "wb") as outfile:
                    for part_file in parts:
                        with open(part_file, "rb") as infile:
                            shutil.copyfileobj(infile, outfile)
                logger.info("Successfully reassembled %s", combined_file.name)
            self._extract_archive(combined_file, self.output_dir / "extracted")

        return results

    @staticmethod
    def _extract_archive(archive_path: Path, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Extracting %s to %s...", archive_path.name, target_dir)
            mode = "r:bz2" if archive_path.name.endswith(".bz2") else "r:gz"
            with tarfile.open(archive_path, mode) as tar:
                tar.extractall(target_dir)
            logger.info("Extracted %s", archive_path.name)
        except Exception as e:
            logger.warning("Extraction note for %s: %s", archive_path.name, e)

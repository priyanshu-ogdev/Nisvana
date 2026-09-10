"""
Project AEGIS — Military Audio Dataset (MAD) Fetcher
Source: Kim, Yoon, & Jung, Scientific Data 11:668 (Nature 2024)
Code repository (annotations + scripts only): https://github.com/kaen2891/military_audio_dataset
Actual audio + train/test split files: Kaggle (junewookim/mad-dataset-military-audio-dataset)
License: CC BY-SA 4.0

VERIFIED 2026-09-03: the GitHub repo's own README states plainly that
"training.csv and test.csv ... must be located in ./data/MAD_dataset/. Please
download the dataset from kaggle." Those files, and all audio, are NOT hosted
in the GitHub repo — only mad_dataset_annotation.csv, README.md, and code are.
The previous version of this fetcher tried to raw-fetch training.csv/test.csv
from GitHub, which 404s. Fixed below: GitHub is used only for the annotation
CSV + README; the actual audio archive is fetched via the Kaggle API, which
requires a free Kaggle account and API token (KAGGLE_USERNAME + KAGGLE_KEY,
or a ~/.kaggle/kaggle.json credentials file — standard Kaggle API setup).
"""

import os
import zipfile
from pathlib import Path
from typing import List
from .base import BaseFetcher, DownloadResult, logger


class MadFetcher(BaseFetcher):
    """
    Fetches the Military Audio Dataset (MAD): annotation/code from GitHub,
    and the actual audio archive (8,075 samples, ~12h, 7 classes, native 48kHz
    mono per the dataset's own Kaggle metadata) via Kaggle API / direct HTTP streaming.
    Supports KAGGLE_ACCESS_TOKEN (Bearer), KAGGLE_USERNAME/KEY (Basic), and ~/.kaggle/kaggle.json.
    """

    BASE_RAW_URL = "https://raw.githubusercontent.com/kaen2891/military_audio_dataset/main"
    # Only files that actually exist in the GitHub repo (verified 2026-09-03).
    # training.csv / test.csv removed — they are not hosted here, see module docstring.
    ANNOTATION_FILES = [
        "mad_dataset_annotation.csv",
        "README.md",
    ]

    KAGGLE_DATASET_SLUG = "junewookim/mad-dataset-military-audio-dataset"
    KAGGLE_API_URL = "https://www.kaggle.com/api/v1/datasets/download/junewookim/mad-dataset-military-audio-dataset"

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results: List[DownloadResult] = []

        # --- Part A: annotation + README from GitHub (small, always attempted) ---
        files = self.ANNOTATION_FILES[:1] if sample_mode else self.ANNOTATION_FILES
        for rel_path in files:
            url = f"{self.BASE_RAW_URL}/{rel_path}"
            dest = self.output_dir / Path(rel_path).name
            res = self.download_file(url, dest, dry_run=dry_run)
            results.append(res)

        # --- Part B: actual audio archive from Kaggle (the ~1.1GB real payload) ---
        kaggle_res = self._fetch_from_kaggle(dry_run=dry_run, sample_mode=sample_mode)
        results.append(kaggle_res)

        return results

    def _fetch_from_kaggle(self, dry_run: bool, sample_mode: bool) -> DownloadResult:
        dest_zip = self.output_dir / "mad_dataset_kaggle.zip"

        # Check if files were manually placed into output directory
        existing_wavs = list(self.output_dir.glob("**/*.wav"))
        if existing_wavs and not dry_run:
            logger.info("Found %d existing MAD audio clips in %s", len(existing_wavs), self.output_dir)
            return DownloadResult(
                success=True,
                destination=existing_wavs[0],
                bytes_downloaded=sum(w.stat().st_size for w in existing_wavs),
                elapsed_sec=0.0,
                md5="",
            )

        if not self._kaggle_credentials_present():
            msg = (
                "Kaggle credentials not configured. The MAD audio archive (~1.1GB, 8,075 clips) "
                "is hosted on Kaggle, not GitHub, and requires Kaggle API credentials. "
                "Set KAGGLE_ACCESS_TOKEN in your .env (recommended, from "
                "https://www.kaggle.com/settings -> API -> Create New Token), or set "
                "KAGGLE_USERNAME and KAGGLE_KEY, or place ~/.kaggle/kaggle.json, then re-run this fetcher. "
                "Only the annotation CSV and README were fetched from GitHub this run."
            )
            if dry_run:
                return DownloadResult(success=False, destination=dest_zip, bytes_downloaded=0, elapsed_sec=0.0, md5="", error=msg)
            logger.warning(msg)
            return DownloadResult(success=False, destination=dest_zip, bytes_downloaded=0, elapsed_sec=0.0, md5="", error=msg)

        if dry_run:
            logger.info(
                "[DRY RUN] Probing MAD audio archive via Kaggle API (%s)...",
                self.KAGGLE_DATASET_SLUG,
            )
            return self.download_file(self.KAGGLE_API_URL, dest_zip, dry_run=True)

        try:
            # Check if archive already downloaded and intact
            if dest_zip.exists() and dest_zip.stat().st_size > 100_000_000:
                logger.info("MAD audio archive %s already exists (%.2f MB).", dest_zip.name, dest_zip.stat().st_size / (1024 * 1024))
                dl_result = DownloadResult(
                    success=True,
                    destination=dest_zip,
                    bytes_downloaded=dest_zip.stat().st_size,
                    elapsed_sec=0.0,
                    md5=self.compute_md5(dest_zip),
                )
            else:
                logger.info("Downloading MAD audio archive from Kaggle (%s)...", self.KAGGLE_API_URL)
                dl_result = self.download_file(self.KAGGLE_API_URL, dest_zip, dry_run=False)
                if not dl_result.success:
                    return dl_result

            # Extract audio clips
            extract_target = self.output_dir / "audio"
            extract_target.mkdir(parents=True, exist_ok=True)
            existing_clips = list(extract_target.glob("*.wav"))
            if not existing_clips and dest_zip.exists():
                logger.info("Extracting MAD audio archive to %s...", extract_target)
                with zipfile.ZipFile(dest_zip, "r") as z:
                    members = [m for m in z.namelist() if m.lower().endswith(".wav")]
                    if sample_mode:
                        members = members[:25]
                    z.extractall(extract_target, members=members)
                logger.info("Extracted %d MAD audio clips to %s", len(members), extract_target)

            return dl_result

        except Exception as e:
            logger.error("Kaggle download failed for %s: %s", self.KAGGLE_DATASET_SLUG, e)
            return DownloadResult(success=False, destination=dest_zip, bytes_downloaded=0, elapsed_sec=0.0, md5="", error=str(e))

    @staticmethod
    def _kaggle_credentials_present() -> bool:
        token = os.environ.get("KAGGLE_ACCESS_TOKEN") or os.environ.get("KAGGLE_API_TOKEN")
        if token and not token.startswith("your_"):
            return True
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        key = os.environ.get("KAGGLE_KEY", "").strip()
        if username and key and not username.startswith("your_") and not key.startswith("your_"):
            return True
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.is_file():
            try:
                import json
                with open(kaggle_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("token") or data.get("access_token") or (data.get("username") and data.get("key")):
                        return True
            except Exception:
                pass
        return False

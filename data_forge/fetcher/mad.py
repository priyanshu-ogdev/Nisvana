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
    mono per the dataset's own Kaggle metadata) via the Kaggle API.
    """

    BASE_RAW_URL = "https://raw.githubusercontent.com/kaen2891/military_audio_dataset/main"
    # Only files that actually exist in the GitHub repo (verified 2026-09-03).
    # training.csv / test.csv removed — they are not hosted here, see module docstring.
    ANNOTATION_FILES = [
        "mad_dataset_annotation.csv",
        "README.md",
    ]

    KAGGLE_DATASET_SLUG = "junewookim/mad-dataset-military-audio-dataset"

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

        if dry_run:
            logger.info(
                "[DRY RUN] MAD audio archive is served via the Kaggle API (dataset: %s), "
                "not a directly probable HTTPS URL. Verifying Kaggle credentials and API "
                "reachability instead of an HTTP HEAD probe.",
                self.KAGGLE_DATASET_SLUG,
            )
            has_creds = self._kaggle_credentials_present()
            return DownloadResult(
                success=has_creds,
                destination=dest_zip,
                bytes_downloaded=0,
                elapsed_sec=0.0,
                md5="",
                error=None if has_creds else (
                    "No Kaggle credentials found (set KAGGLE_USERNAME + KAGGLE_KEY env vars, "
                    "or place ~/.kaggle/kaggle.json). Required to fetch the real MAD audio archive."
                ),
            )

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
                "is hosted on Kaggle, not GitHub, and requires a free Kaggle account + API token. "
                "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables (from "
                "https://www.kaggle.com/settings -> API -> Create New Token), or place the "
                "downloaded kaggle.json at ~/.kaggle/kaggle.json, then re-run this fetcher. "
                "Only the annotation CSV and README were fetched from GitHub this run."
            )
            logger.warning(msg)
            return DownloadResult(success=False, destination=dest_zip, bytes_downloaded=0, elapsed_sec=0.0, md5="", error=msg)

        try:
            # Imported lazily: kaggle's client reads credentials from env/file at import time,
            # so this must happen only after we've confirmed credentials exist above.
            import kaggle  # type: ignore

            logger.info("Authenticating with Kaggle API and downloading dataset '%s'...", self.KAGGLE_DATASET_SLUG)
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                self.KAGGLE_DATASET_SLUG,
                path=str(self.output_dir),
                unzip=False,
                quiet=False,
            )
            downloaded = list(self.output_dir.glob("*.zip"))
            if not downloaded:
                raise RuntimeError("Kaggle API reported success but no .zip archive was found in the output directory.")
            archive = downloaded[0]

            if not dry_run:
                extract_target = self.output_dir / "audio"
                if not extract_target.exists():
                    logger.info("Extracting MAD audio archive...")
                    with zipfile.ZipFile(archive, "r") as z:
                        members = z.namelist()
                        if sample_mode:
                            members = [m for m in members if m.lower().endswith(".wav")][:25]
                        z.extractall(extract_target, members=members)

            size = archive.stat().st_size
            return DownloadResult(success=True, destination=archive, bytes_downloaded=size, elapsed_sec=0.0, md5=self.compute_md5(archive))

        except Exception as e:
            logger.error("Kaggle download failed for %s: %s", self.KAGGLE_DATASET_SLUG, e)
            return DownloadResult(success=False, destination=dest_zip, bytes_downloaded=0, elapsed_sec=0.0, md5="", error=str(e))

    @staticmethod
    def _kaggle_credentials_present() -> bool:
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True
        return (Path.home() / ".kaggle" / "kaggle.json").exists()

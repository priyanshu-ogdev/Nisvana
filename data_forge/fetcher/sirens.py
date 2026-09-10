"""
Project AEGIS — Sirens and Urban Hazard Audio Fetcher
Sources: UrbanSound8K (Salamon et al.) & ESC-50 (Piczak)
"""

import zipfile
from pathlib import Path
from typing import List
from .base import BaseFetcher, DownloadResult, logger


class SirensFetcher(BaseFetcher):
    """
    Fetches siren emergency audio and environmental wind recordings from ESC-50 and UrbanSound.
    Supports individual raw clip streaming, Zenodo archive mirror, and GitHub master zip fallbacks.
    """

    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio"
    ZENODO_ARCHIVE_URL = "https://zenodo.org/records/2452749/files/ESC-50-master.zip"
    GITHUB_ARCHIVE_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    
    # Selected verified siren and environmental clips from ESC-50
    # Category 42 = Siren, Category 39 = Wind
    SAMPLE_CLIPS = [
        "1-31482-A-42.wav",  # Siren
        "1-31482-B-42.wav",  # Siren
        "1-54084-A-42.wav",  # Siren
        "1-76831-A-42.wav",  # Siren
        "1-76831-B-42.wav",  # Siren
        "1-20133-A-39.wav",  # Wind
        "1-84536-A-39.wav",  # Wind
        "1-84704-A-39.wav",  # Wind
    ]

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results: List[DownloadResult] = []

        # Check if files were already downloaded or extracted locally
        existing_wavs = list(self.output_dir.glob("*.wav"))
        if existing_wavs and not dry_run:
            logger.info("Found %d existing siren/wind clips in %s", len(existing_wavs), self.output_dir)
            return [
                DownloadResult(
                    success=True,
                    destination=w,
                    bytes_downloaded=w.stat().st_size,
                    elapsed_sec=0.0,
                    md5="",
                )
                for w in existing_wavs
            ]

        if sample_mode or dry_run:
            clips = self.SAMPLE_CLIPS[:3] if sample_mode else self.SAMPLE_CLIPS
            for clip_name in clips:
                primary_url = f"{self.GITHUB_RAW_BASE}/{clip_name}"
                fallback_urls = [
                    f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{clip_name}",
                    f"https://github.com/karolpiczak/ESC-50/raw/master/audio/{clip_name}",
                ]
                dest = self.output_dir / clip_name
                res = self.download_file(primary_url, dest, dry_run=dry_run, fallback_urls=fallback_urls)
                results.append(res)
            return results

        # Full production mode: download complete ESC-50 archive from Zenodo (or GitHub zip fallback)
        # and extract all Siren (category 42) and Wind (category 39) files
        dest_zip = self.output_dir / "ESC-50-master.zip"
        res_zip = self.download_file(
            self.ZENODO_ARCHIVE_URL,
            dest_zip,
            dry_run=False,
            fallback_urls=[self.GITHUB_ARCHIVE_URL],
        )
        results.append(res_zip)

        if res_zip.success and dest_zip.exists():
            try:
                logger.info("Extracting siren (42) and wind (39) audio clips from %s...", dest_zip.name)
                with zipfile.ZipFile(dest_zip, "r") as z:
                    # ESC-50 naming: {fold}-{clip_id}-{take}-{class_id}.wav
                    target_members = [
                        m for m in z.namelist()
                        if m.lower().endswith(".wav") and (m.rsplit("-", 1)[-1].split(".")[0] in ("42", "39"))
                    ]
                    for member in target_members:
                        filename = Path(member).name
                        target_file = self.output_dir / filename
                        if not target_file.exists():
                            with z.open(member) as src, open(target_file, "wb") as dst:
                                dst.write(src.read())
                logger.info("Successfully extracted %d siren and wind audio clips.", len(target_members))
            except Exception as e:
                logger.warning("Error unpacking ESC-50 archive: %s", e)

        return results

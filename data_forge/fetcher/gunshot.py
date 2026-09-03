import os
from pathlib import Path
from typing import List
from .base import BaseFetcher, DownloadResult, logger


class GunshotDryadFetcher(BaseFetcher):
    """
    Fetches real multi-firearm field gunshot recordings from Dryad repository.
    Includes 7 firearms (pistols and rifles) recorded with multiple synchronized microphones.
    DOI: 10.5061/dryad.wm37pvmkc
    """

    BASE_API_URL = "https://datadryad.org/api/v2/files"
    FILES = [
        ("398394", "readme.txt"),
        ("398398", "mic1raw.wav"),     # Main synchronized microphone
        ("398397", "mic2raw16b.wav"),  # Secondary microphone
    ]

    def __init__(self, output_dir: Path, timeout: int = 60, max_retries: int = 5):
        super().__init__(output_dir, timeout, max_retries)
        dryad_token = os.environ.get("DRYAD_API_TOKEN") or os.environ.get("DRYAD_TOKEN")
        if dryad_token:
            self.headers["Authorization"] = f"Bearer {dryad_token}"

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results = []
        files_to_download = self.FILES[:1] if sample_mode else self.FILES

        # Check if files were manually placed into output directory
        existing_wavs = list(self.output_dir.glob("*.wav"))
        if existing_wavs and not dry_run:
            logger.info("Found %d existing gunshot WAV files in %s", len(existing_wavs), self.output_dir)
            return [DownloadResult(success=True, destination=w, bytes_downloaded=w.stat().st_size, elapsed_sec=0.0, md5="") for w in existing_wavs]

        for file_id, filename in files_to_download:
            url = f"{self.BASE_API_URL}/{file_id}/download"
            dest = self.output_dir / filename
            res = self.download_file(url, dest, dry_run=dry_run)
            if not res.success and "401" in (res.error or ""):
                logger.info(
                    "Dryad API requires a free bearer token for downloads. "
                    "Set DRYAD_API_TOKEN environment variable or place '%s' directly into %s",
                    filename,
                    self.output_dir,
                )
            results.append(res)

        return results

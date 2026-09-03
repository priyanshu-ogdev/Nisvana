"""
Project AEGIS — Sirens and Urban Hazard Audio Fetcher
Sources: UrbanSound8K (Salamon et al.) & ESC-50 (Piczak)
"""

from pathlib import Path
from typing import List
from .base import BaseFetcher, DownloadResult, logger


class SirensFetcher(BaseFetcher):
    """
    Fetches siren emergency audio and thin urban classes (wind, rotor clips).
    """

    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio"
    
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
        results = []
        clips = self.SAMPLE_CLIPS[:3] if sample_mode else self.SAMPLE_CLIPS

        for clip_name in clips:
            url = f"{self.GITHUB_RAW_BASE}/{clip_name}"
            dest = self.output_dir / clip_name
            res = self.download_file(url, dest, dry_run=dry_run)
            results.append(res)

        return results

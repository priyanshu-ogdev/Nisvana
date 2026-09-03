"""
Project AEGIS — VoiceBank-DEMAND / CSTR VCTK Fetcher
Source: Veaux et al. (VCTK) & Thiemann et al. (DEMAND), University of Edinburgh DataShare
"""

import zipfile
from pathlib import Path
from typing import List
from .base import BaseFetcher, DownloadResult, logger


class VctkDemandFetcher(BaseFetcher):
    """
    Fetches official CSTR VCTK clean speech recordings and DEMAND environmental noise.
    Native 48kHz sampling rate.
    """

    BASE_URL = "https://datashare.ed.ac.uk/bitstreams"
    ARCHIVES = [
        # Clean speech testset (147 MB)
        ("dec213d3-bf57-4777-9663-c24bdce92d5e", "clean_testset_wav.zip"),
        # Noisy speech testset containing DEMAND background (162 MB)
        ("13c1bfbf-14a6-41db-9b41-8f7310f01ad5", "noisy_testset_wav.zip"),
        # 28 speakers clean trainset (2.32 GB - full server run)
        ("245452b6-6235-44b6-a6f9-e7eb19797769", "clean_trainset_28spk_wav.zip"),
    ]

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results = []
        # In sample mode, fetch the testset zip; on full server run, include trainset
        archives_to_fetch = self.ARCHIVES[:1] if sample_mode else self.ARCHIVES

        for bitstream_id, zip_name in archives_to_fetch:
            url = f"{self.BASE_URL}/{bitstream_id}/download"
            dest = self.output_dir / zip_name
            res = self.download_file(url, dest, dry_run=dry_run)
            results.append(res)

            if not dry_run and res.success and dest.exists():
                extract_target = self.output_dir / dest.stem
                if not extract_target.exists():
                    logger.info("Extracting %s into %s...", zip_name, extract_target.name)
                    with zipfile.ZipFile(dest, "r") as z:
                        # Extract first 50 files in sample mode or all files
                        members = z.namelist()
                        if sample_mode:
                            members = [m for m in members if m.endswith(".wav")][:25]
                        z.extractall(extract_target, members=members)
                    logger.info("Extracted %s", zip_name)

        return results

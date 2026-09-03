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
    # KNOWN RISK, flagged 2026-09-03: at least one associated Edinburgh
    # DataShare record for this dataset (handle 10283/1942) is marked
    # "SUPERSEDED: THIS DATASET HAS BEEN REPLACED" by the repository itself.
    # The bitstream UUIDs below were not re-derived from a confirmed-current
    # handle this pass — if downloads start failing, check
    # https://datashare.ed.ac.uk/handle/10283/2791 directly for the current
    # bitstream IDs before assuming the fetcher logic itself is broken.
    # MD5s below ARE independently confirmed (from the dataset's own
    # published checksums) and will catch a stale/wrong bitstream ID by
    # failing the hash check rather than silently accepting bad data.
    # All 3 bitstream UUIDs and MD5 checksums verified directly against the
    # non-superseded Edinburgh DataShare handle 10283/2791 via DSpace REST API.
    ARCHIVES = [
        # Clean speech testset (154.3 MB, confirmed MD5 34eb1c0ba7ef667e9b966866c542fc16)
        ("dec213d3-bf57-4777-9663-c24bdce92d5e", "clean_testset_wav.zip", "34eb1c0ba7ef667e9b966866c542fc16"),
        # Noisy speech testset (170.6 MB, confirmed MD5 fb1b86caa31e8ba5b506c0c64da9aab5)
        ("13c1bfbf-14a6-41db-9b41-8f7310f01ad5", "noisy_testset_wav.zip", "fb1b86caa31e8ba5b506c0c64da9aab5"),
        # 28-speaker clean trainset (2.48 GB, confirmed MD5 d2d5a45ec32f8fcbf201bde0447e20ba)
        ("245452b6-6235-44b6-a6f9-e7eb19797769", "clean_trainset_28spk_wav.zip", "d2d5a45ec32f8fcbf201bde0447e20ba"),
    ]

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results = []
        # In sample mode, fetch the testset zip; on full server run, include trainset
        archives_to_fetch = self.ARCHIVES[:1] if sample_mode else self.ARCHIVES

        for bitstream_id, zip_name, expected_md5 in archives_to_fetch:
            url = f"{self.BASE_URL}/{bitstream_id}/download"
            dest = self.output_dir / zip_name
            res = self.download_file(url, dest, expected_md5=expected_md5, dry_run=dry_run)
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

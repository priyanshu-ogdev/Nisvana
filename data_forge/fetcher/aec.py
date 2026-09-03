"""
Project AEGIS — AEC Challenge Fetcher
Source: Microsoft ICASSP Acoustic Echo Cancellation Challenge
"""

import zipfile
from pathlib import Path
from typing import List
from .base import BaseFetcher, DownloadResult, logger


class AecChallengeFetcher(BaseFetcher):
    """
    Fetches official AEC Challenge fullband 48kHz dataset.
    Preserves far-end, near-end, echo, and microphone coupling intact.
    """

    FULLBAND_SYNTHETIC_URL = "https://aecchallengepublic.blob.core.windows.net/icassp2022/fullband_synthetic.zip"
    EXPECTED_MD5 = "4016e887707f5570069960ebf263b644"

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        dest = self.output_dir / "fullband_synthetic.zip"
        res = self.download_file(self.FULLBAND_SYNTHETIC_URL, dest, expected_md5=None if sample_mode else self.EXPECTED_MD5, dry_run=dry_run)

        if not dry_run and res.success and dest.exists():
            extract_target = self.output_dir / "aec_pairs"
            if not extract_target.exists():
                logger.info("Extracting AEC pairs (mic, farend, nearend, echo)...")
                with zipfile.ZipFile(dest, "r") as z:
                    wav_files = [m for m in z.namelist() if m.endswith(".wav")]
                    if sample_mode:
                        # Extract 5 matched sets (_mic, _farend, _nearend, _echo)
                        sample_prefixes = set([w.rsplit("_", 1)[0] for w in wav_files[:20]])
                        wav_files = [w for w in wav_files if any(w.startswith(p) for p in list(sample_prefixes)[:5])]
                    z.extractall(extract_target, members=wav_files)
                logger.info("Extracted %d AEC challenge files", len(wav_files))

        return [res]

"""
Project AEGIS — AEC Challenge Fetcher
Source: Microsoft ICASSP Acoustic Echo Cancellation Challenge
Repository: https://github.com/microsoft/AEC-Challenge

VERIFIED 2026-09-03:
- Primary Endpoint: Official Microsoft AEC-Challenge Git LFS repository:
  https://media.githubusercontent.com/media/microsoft/AEC-Challenge/main/datasets/synthetic/
- Quadruplets: (mic, farend, nearend, echo) representing authentic acoustic echo path physics.
- Fallback / Drop-in: fullband_synthetic.zip or pre-placed WAV files in data/raw/aec_challenge/
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

    GITHUB_LFS_BASE = "https://media.githubusercontent.com/media/microsoft/AEC-Challenge/main/datasets/synthetic"
    SAMPLE_FILE_IDS = [0, 1, 2, 3, 4]  # 5 complete matched quadruplets (20 WAVs) for sample mode

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results: List[DownloadResult] = []
        extract_target = self.output_dir / "aec_pairs"
        extract_target.mkdir(parents=True, exist_ok=True)

        # Check if files already exist locally
        existing_wavs = list(self.output_dir.glob("**/*.wav"))
        if existing_wavs and not dry_run:
            logger.info("Found %d existing AEC Challenge WAV files in %s", len(existing_wavs), self.output_dir)
            return [
                DownloadResult(
                    success=True,
                    destination=existing_wavs[0],
                    bytes_downloaded=sum(w.stat().st_size for w in existing_wavs),
                    elapsed_sec=0.0,
                    md5="",
                )
            ]

        # Check if fullband_synthetic.zip was dropped in manually
        zip_path = self.output_dir / "fullband_synthetic.zip"
        if zip_path.exists() and not dry_run:
            logger.info("Extracting manual fullband_synthetic.zip drop-in...")
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    wav_members = [m for m in z.namelist() if m.endswith(".wav")]
                    if sample_mode:
                        sample_prefixes = set([w.rsplit("_", 1)[0] for w in wav_files[:20]])
                        wav_members = [w for w in wav_members if any(w.startswith(p) for p in list(sample_prefixes)[:5])]
                    z.extractall(extract_target, members=wav_members)
                logger.info("Extracted %d AEC challenge files from zip.", len(wav_members))
                return [DownloadResult(success=True, destination=zip_path, bytes_downloaded=zip_path.stat().st_size, elapsed_sec=0.0, md5="")]
            except Exception as e:
                logger.warning("Error unpacking manual zip: %s", e)

        # Fetch authentic synthetic quadruplets from official Microsoft repository
        file_ids = self.SAMPLE_FILE_IDS if sample_mode else list(range(100))  # 100 sets = 400 WAV files for training

        if dry_run:
            # Probe first far-end and mic files
            probe_url = f"{self.GITHUB_LFS_BASE}/farend_speech/farend_speech_fileid_0.wav"
            res = self.download_file(probe_url, extract_target / "synthetic_fileid_0_farend.wav", dry_run=True)
            results.append(res)
            return results

        # Download matched quadruplets
        for fid in file_ids:
            quadruplets = [
                ("farend_speech", f"farend_speech_fileid_{fid}.wav", f"synthetic_fileid_{fid}_farend.wav"),
                ("nearend_speech", f"nearend_speech_fileid_{fid}.wav", f"synthetic_fileid_{fid}_nearend.wav"),
                ("nearend_mic_signal", f"nearend_mic_fileid_{fid}.wav", f"synthetic_fileid_{fid}_mic.wav"),
                ("echo_signal", f"echo_fileid_{fid}.wav", f"synthetic_fileid_{fid}_echo.wav"),
            ]
            for folder, remote_name, local_name in quadruplets:
                url = f"{self.GITHUB_LFS_BASE}/{folder}/{remote_name}"
                dest = extract_target / local_name
                r = self.download_file(url, dest, dry_run=False)
                results.append(r)

        return results

"""
Project AEGIS — DroneAudioSet Fetcher
Source: ahlab-drone-project / Augmented Human Lab (Hugging Face)
License: MIT
"""

import io
from pathlib import Path
from typing import List
import pandas as pd
import scipy.io.wavfile as wavfile
from .base import BaseFetcher, DownloadResult, logger


class DroneAudioSetFetcher(BaseFetcher):
    """
    Fetches real UAV ego-noise, propeller, and multi-rotor audio from DroneAudioSet.
    """

    BASE_RESOLVE_URL = "https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/resolve/main"
    PARQUET_FILES = [
        "drone-only/train_001-00000-of-00001.parquet",
        "drone-only/train_002-00000-of-00001.parquet",
    ]

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results = []
        files = self.PARQUET_FILES[:1] if sample_mode else self.PARQUET_FILES

        extracted_dir = self.output_dir / "wavs"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        for rel_path in files:
            url = f"{self.BASE_RESOLVE_URL}/{rel_path}"
            filename = Path(rel_path).name
            dest = self.output_dir / filename
            res = self.download_file(url, dest, dry_run=dry_run)
            results.append(res)

            if not dry_run and res.success and dest.exists():
                try:
                    logger.info("Extracting drone WAV samples from %s...", dest.name)
                    df = pd.read_parquet(dest)
                    # Check audio column
                    if "audio" in df.columns:
                        for idx, row in df.iterrows():
                            if sample_mode and idx >= 15:
                                break
                            audio_data = row["audio"]
                            if isinstance(audio_data, dict):
                                audio_bytes = audio_data.get("bytes")
                                sr = audio_data.get("sampling_rate", 48000)
                                if audio_bytes:
                                    out_wav = extracted_dir / f"drone_{filename[:9]}_{idx:04d}.wav"
                                    with open(out_wav, "wb") as wf:
                                        wf.write(audio_bytes)
                    logger.info("Extracted drone audio clips from %s", dest.name)
                except Exception as e:
                    logger.warning("Parquet extraction note: %s", e)

        return results

"""
Project AEGIS — SHAReD Dataset Fetcher (Explosion / Artillery Audio)
Source: Takazawa et al., Sensors 2024 / Harvard Dataverse (doi:10.7910/DVN/ROWODP)
"""

import pickle
from pathlib import Path
from typing import List
import numpy as np
import scipy.io.wavfile as wavfile
from .base import BaseFetcher, DownloadResult, logger


class SharedExplosionFetcher(BaseFetcher):
    """
    Fetches the Smartphone High-Explosive Audio Recordings Dataset (SHAReD).
    Contains 326 high-explosive blast waveforms.
    """

    DATAVERSE_FILE_ID = "10192135"  # SHAReD.pkl Dataverse ID
    DOWNLOAD_URL = f"https://dataverse.harvard.edu/api/access/datafile/{DATAVERSE_FILE_ID}"

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        pkl_dest = self.output_dir / "SHAReD.pkl"
        res = self.download_file(self.DOWNLOAD_URL, pkl_dest, dry_run=dry_run)

        if dry_run or not res.success:
            return [res]

        # If downloaded, extract the waveforms to individual WAV files
        extracted_dir = self.output_dir / "wavs"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Unpacking SHAReD blast waveforms from %s...", pkl_dest.name)
            with open(pkl_dest, "rb") as f:
                data = pickle.load(f)

            # SHAReD structure contains dictionary or list of waveforms and sample rate (typically 48kHz or 44.1kHz)
            # We extract them to standard WAV files
            count = 0
            if isinstance(data, dict):
                waveforms = data.get("audio", data.get("waveforms", data.get("data", [])))
                sr = data.get("samplerate", data.get("sr", 48000))
                if isinstance(waveforms, list) or isinstance(waveforms, np.ndarray):
                    for idx, wf in enumerate(waveforms):
                        if sample_mode and idx >= 10:
                            break
                        wf_arr = np.asarray(wf, dtype=np.float32)
                        # Normalize float range to [-1, 1]
                        max_abs = np.max(np.abs(wf_arr)) if len(wf_arr) > 0 else 1.0
                        if max_abs > 0:
                            wf_norm = (wf_arr / max_abs * 32767).astype(np.int16)
                        else:
                            wf_norm = wf_arr.astype(np.int16)
                        out_path = extracted_dir / f"shared_blast_{idx:03d}.wav"
                        wavfile.write(out_path, int(sr), wf_norm)
                        count += 1
            logger.info("Extracted %d SHAReD explosion waveforms into %s", count, extracted_dir)
        except Exception as e:
            logger.warning("SHAReD pickle extraction deferred or format varied: %s", e)

        return [res]

"""
Project AEGIS — SHAReD Dataset Fetcher (Explosion / Artillery Audio)
Source: Takazawa et al., Sensors 2024 / Harvard Dataverse (doi:10.7910/DVN/ROWODP)
"""

import pickle
from pathlib import Path
from typing import Any, List
import numpy as np
import scipy.io.wavfile as wavfile
from .base import BaseFetcher, DownloadResult, logger


def _save_waveform(wf: Any, sr: int, out_path: Path) -> bool:
    """Safely normalizes and writes audio array/bytes as standard 16-bit PCM WAV."""
    try:
        if isinstance(wf, bytes):
            out_path.write_bytes(wf)
            return True

        arr = np.asarray(wf, dtype=np.float32).squeeze()
        if arr.size == 0:
            return False

        # Clean NaNs and infinite values
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

        if arr.ndim > 1:
            # Downmix multi-channel to mono
            axis = 0 if arr.shape[0] < arr.shape[1] else 1
            arr = np.mean(arr, axis=axis)

        # Center waveform (remove DC offset)
        if len(arr) > 0:
            arr = arr - np.mean(arr)

        max_abs = np.max(np.abs(arr))
        if max_abs > 1.0:
            # Scale if data is already in integer range or unnormalized
            arr = arr / max_abs

        norm_int16 = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(out_path, int(sr), norm_int16)
        return True
    except Exception as e:
        logger.debug("Failed saving waveform to %s: %s", out_path.name, e)
        return False


class SharedExplosionFetcher(BaseFetcher):
    """
    Fetches the Smartphone High-Explosive Audio Recordings Dataset (SHAReD).
    Contains 326 high-explosive blast waveforms.
    """

    DATAVERSE_FILE_ID = "10192135"  # SHAReD.pkl Dataverse ID
    DOWNLOAD_URL = f"https://dataverse.harvard.edu/api/access/datafile/{DATAVERSE_FILE_ID}"

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        # Check if WAV files were already extracted (expecting >= 300 waveforms for full dataset)
        existing_wavs = list(self.output_dir.glob("**/*.wav"))
        min_expected = 10 if sample_mode else 300
        if len(existing_wavs) >= min_expected and not dry_run:
            logger.info("Found %d existing SHAReD blast WAV files in %s", len(existing_wavs), self.output_dir)
            return [DownloadResult(success=True, destination=w, bytes_downloaded=w.stat().st_size, elapsed_sec=0.0, md5="") for w in existing_wavs]

        pkl_dest = self.output_dir / "SHAReD.pkl"

        # Check if SHAReD.pkl was manually placed into output directory
        if pkl_dest.exists() and pkl_dest.stat().st_size > 0 and not dry_run:
            logger.info("Found manually placed SHAReD.pkl in %s (%.2f MB). Unpacking waveforms...", self.output_dir, pkl_dest.stat().st_size / (1024 * 1024))
            res = DownloadResult(success=True, destination=pkl_dest, bytes_downloaded=pkl_dest.stat().st_size, elapsed_sec=0.0, md5="")
        else:
            res = self.download_file(self.DOWNLOAD_URL, pkl_dest, dry_run=dry_run)

        if dry_run or not res.success:
            if not res.success:
                logger.info(
                    "Harvard Dataverse download failed or requires credentials. "
                    "You can manually download SHAReD.pkl from: %s (or %s) and place it directly into %s",
                    self.DOWNLOAD_URL,
                    "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ROWODP",
                    self.output_dir,
                )
            return [res]

        # If downloaded or manually placed, extract the waveforms to individual WAV files
        extracted_dir = self.output_dir / "wavs"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Unpacking SHAReD blast waveforms from %s...", pkl_dest.name)
            with open(pkl_dest, "rb") as f:
                data = pickle.load(f)

            count = 0
            default_sr = 48000
            data_type = type(data).__name__
            logger.info("Loaded SHAReD.pkl object: type=%s", data_type)

            # --- Case 1: Pandas DataFrame or Series ---
            if hasattr(data, "columns") or hasattr(data, "to_dict"):
                try:
                    import pandas as pd
                    if isinstance(data, pd.DataFrame):
                        logger.info("SHAReD DataFrame shape=%s, columns=%s", data.shape, list(data.columns)[:10])
                        # Prioritize microphone_data (exact column in Takazawa et al. Harvard Dataverse)
                        audio_col = next(
                            (col for col in data.columns if col in [
                                "microphone_data", "microphone", "mic_data", "audio",
                                "waveform", "waveforms", "signal", "signals", "data",
                                "wav", "x", "recording", "clip"
                            ] or "microphone_data" in col or "mic_data" in col or
                            ("microphone" in col and "time" not in col and "rate" not in col)),
                            None
                        )
                        sr_col = next(
                            (col for col in data.columns if col in [
                                "microphone_sample_rate_hz", "microphone_sample_rate",
                                "sample_rate", "samplerate", "sampling_rate", "sr", "fs"
                            ] or ("sample_rate" in col and "microphone" in col) or
                            ("sample_rate" in col and "barometer" not in col and "accelerometer" not in col)),
                            None
                        )

                        if audio_col:
                            logger.info("Found audio column '%s' and sample-rate column '%s'", audio_col, sr_col)
                            for idx, row in data.iterrows():
                                if sample_mode and idx >= 10:
                                    break
                                val = row[audio_col]
                                sr = row[sr_col] if sr_col and row[sr_col] and not pd.isna(row[sr_col]) else default_sr
                                if isinstance(val, dict):
                                    wf = val.get("bytes", val.get("array", val.get("data", val.get("audio"))))
                                    sr = val.get("sampling_rate", val.get("sample_rate", sr))
                                else:
                                    wf = val
                                out_p = extracted_dir / f"shared_blast_{idx:03d}.wav"
                                if _save_waveform(wf, int(sr), out_p):
                                    count += 1
                        else:
                            # Matrix of time-series: only if large number of samples per column/row
                            if data.shape[0] < data.shape[1] and data.shape[1] > 1000:
                                for idx in range(data.shape[0]):
                                    if sample_mode and idx >= 10:
                                        break
                                    out_p = extracted_dir / f"shared_blast_{idx:03d}.wav"
                                    if _save_waveform(data.iloc[idx].to_numpy(), default_sr, out_p):
                                        count += 1
                            elif data.shape[1] <= 1000 and data.shape[0] > 1000:
                                for idx, col in enumerate(data.columns):
                                    if sample_mode and idx >= 10:
                                        break
                                    out_p = extracted_dir / f"shared_blast_{idx:03d}.wav"
                                    if _save_waveform(data[col].to_numpy(), default_sr, out_p):
                                        count += 1
                except Exception as df_err:
                    logger.debug("DataFrame unpack branch note: %s", df_err)

            # --- Case 2: Python Dictionary ---
            if count == 0 and isinstance(data, dict):
                logger.info("SHAReD dictionary keys: %s", list(data.keys())[:15])
                sr = data.get("samplerate", data.get("sample_rate", data.get("sr", data.get("fs", default_sr))))

                # Check known audio array keys
                container_key = next((k for k in ["audio", "waveform", "waveforms", "signals", "signal", "data", "recordings", "x", "X", "blasts", "clips", "samples"] if k in data), None)

                if container_key:
                    items = data[container_key]
                    if isinstance(items, (list, tuple, np.ndarray)):
                        for idx, item in enumerate(items):
                            if sample_mode and idx >= 10:
                                break
                            out_p = extracted_dir / f"shared_blast_{idx:03d}.wav"
                            if _save_waveform(item, int(sr), out_p):
                                count += 1
                else:
                    # Check split dictionaries e.g. {'train': [...], 'test': [...]}
                    for split_name in ["train", "test", "val", "eval"]:
                        if split_name in data and isinstance(data[split_name], (list, tuple, np.ndarray, dict)):
                            sub = data[split_name]
                            if isinstance(sub, dict):
                                for k_sub, v_sub in sub.items():
                                    if sample_mode and count >= 10:
                                        break
                                    out_p = extracted_dir / f"shared_blast_{count:03d}.wav"
                                    if _save_waveform(v_sub, int(sr), out_p):
                                        count += 1
                            else:
                                for item in sub:
                                    if sample_mode and count >= 10:
                                        break
                                    out_p = extracted_dir / f"shared_blast_{count:03d}.wav"
                                    if _save_waveform(item, int(sr), out_p):
                                        count += 1

                    # Check if dict is mapping of clip_id -> waveform or clip_id -> dict
                    if count == 0:
                        for k, v in data.items():
                            if sample_mode and count >= 10:
                                break
                            clean_k = str(k).replace("/", "_").replace("\\", "_")
                            if isinstance(v, dict):
                                wf = v.get("audio", v.get("waveform", v.get("signal", v.get("data", v.get("array")))))
                                clip_sr = v.get("sr", v.get("samplerate", v.get("sample_rate", sr)))
                                if wf is not None:
                                    out_p = extracted_dir / f"shared_{clean_k}.wav"
                                    if _save_waveform(wf, int(clip_sr), out_p):
                                        count += 1
                            elif isinstance(v, (list, tuple, np.ndarray)):
                                out_p = extracted_dir / f"shared_{clean_k}.wav"
                                if _save_waveform(v, int(sr), out_p):
                                    count += 1

            # --- Case 3: List or Tuple ---
            if count == 0 and isinstance(data, (list, tuple)):
                logger.info("SHAReD list/tuple container with %d elements", len(data))
                for idx, item in enumerate(data):
                    if sample_mode and idx >= 10:
                        break
                    sr = default_sr
                    if isinstance(item, dict):
                        wf = item.get("audio", item.get("waveform", item.get("signal", item.get("data", item.get("array")))))
                        sr = item.get("sr", item.get("samplerate", item.get("sample_rate", sr)))
                    elif isinstance(item, tuple) and len(item) == 2:
                        wf, potential_sr = item
                        if isinstance(potential_sr, (int, float)) and potential_sr > 1000:
                            sr = potential_sr
                    else:
                        wf = item

                    if wf is not None:
                        out_p = extracted_dir / f"shared_blast_{idx:03d}.wav"
                        if _save_waveform(wf, int(sr), out_p):
                            count += 1

            # --- Case 4: NumPy ndarray ---
            if count == 0 and isinstance(data, np.ndarray):
                logger.info("SHAReD NumPy array shape=%s, dtype=%s", data.shape, data.dtype)
                if data.ndim == 2:
                    if data.shape[0] <= data.shape[1]:
                        for idx in range(data.shape[0]):
                            if sample_mode and idx >= 10:
                                break
                            out_p = extracted_dir / f"shared_blast_{idx:03d}.wav"
                            if _save_waveform(data[idx], default_sr, out_p):
                                count += 1
                    else:
                        for idx in range(data.shape[1]):
                            if sample_mode and idx >= 10:
                                break
                            out_p = extracted_dir / f"shared_blast_{idx:03d}.wav"
                            if _save_waveform(data[:, idx], default_sr, out_p):
                                count += 1
                elif data.ndim == 1:
                    out_p = extracted_dir / "shared_blast_000.wav"
                    if _save_waveform(data, default_sr, out_p):
                        count += 1

            logger.info("Extracted %d SHAReD explosion waveforms into %s", count, extracted_dir)
        except Exception as e:
            logger.warning("SHAReD pickle extraction deferred or format varied: %s", e)

        # Return updated list of extracted WAV files if any were produced
        extracted_wavs = list(extracted_dir.glob("*.wav"))
        if extracted_wavs:
            return [DownloadResult(success=True, destination=w, bytes_downloaded=w.stat().st_size, elapsed_sec=0.0, md5="") for w in extracted_wavs]

        return [res]

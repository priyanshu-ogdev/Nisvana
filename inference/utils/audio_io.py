"""
inference/utils/audio_io.py — Edge Audio I/O Utilities

Loads and saves 48,000 Hz mono audio with clipping prevention,
peak normalization, and multi-channel downmixing.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from scipy import signal
import soundfile as sf
import torch


def load_audio_48k(
    file_path: Union[str, Path],
    target_sr: int = 48000,
    normalize: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Loads audio file, downmixes to mono, and resamples to target_sr (48000 Hz).
    Returns:
        (mono_audio_float32, sample_rate)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)

    # Downmix to mono: average channels
    mono = np.mean(data, axis=1)

    # Resample if sample rate doesn't match
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        mono = signal.resample_poly(mono, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr

    if normalize:
        peak = np.max(np.abs(mono))
        if peak > 1e-6:
            mono = mono / peak * 0.95

    return mono, sr


def save_audio_48k(
    file_path: Union[str, Path],
    audio: Union[np.ndarray, torch.Tensor],
    sr: int = 48000,
    subtype: str = "PCM_16",
) -> Path:
    """
    Saves audio to disk with clipping prevention [-1.0, 1.0].
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    audio = np.ascontiguousarray(audio.squeeze(), dtype=np.float32)

    # Prevent hard digital wrap-around clipping
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(str(file_path), audio, samplerate=sr, subtype=subtype)
    return file_path

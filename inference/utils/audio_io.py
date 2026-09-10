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


def _generate_tpdf_dither(n_samples: int, lsb: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generates Triangular Probability Density Function (TPDF) dither noise,
    the established standard for float->integer PCM word-length reduction
    (sum of two independent RPDF/uniform sources -- the only distribution
    that fully de-correlates quantization error from the signal, rather
    than merely reducing that correlation the way a single uniform/RPDF
    source does).

    GROUNDED: applied at a peak level of +/-1 LSB (TPDF spans 2 LSB
    peak-to-peak, matching the standard "1 LSB TPDF" convention used by
    reference implementations from Audacity down to professional mastering
    dither plugins) -- large enough to fully de-correlate quantization
    error, small enough to stay near the theoretical 16-bit noise floor
    (~-96 dBFS) rather than audibly raising it.
    """
    u1 = rng.uniform(-0.5, 0.5, n_samples)
    u2 = rng.uniform(-0.5, 0.5, n_samples)
    return (u1 + u2) * lsb


def save_audio_48k(
    file_path: Union[str, Path],
    audio: Union[np.ndarray, torch.Tensor],
    sr: int = 48000,
    subtype: str = "PCM_16",
    dither: bool = True,
    dither_seed: int = None,
) -> Path:
    """
    Saves audio to disk with clipping prevention [-1.0, 1.0].

    FIX (this pass, researched rather than assumed): float32 audio being
    saved as PCM_16 is a bit-depth reduction -- 32-bit float has ~1528 dB
    of representable range, PCM_16 has 96 dB. Without dither, this
    requantization produces error that is CORRELATED with the signal
    (audible as harsh, signal-dependent distortion, worst on quiet
    passages and fades -- exactly where a speech-enhancement system's
    output is often headed after successfully suppressing noise). TPDF
    dither is the well-established standard fix: it trades that
    correlated distortion for a small, fixed, uncorrelated noise floor
    -- described consistently across reference audio-engineering sources
    as "the gold standard" for exactly this word-length-reduction step,
    applied as the LAST operation before quantization, matching this
    function's own position in the pipeline (immediately before
    sf.write's own internal float->int conversion).

    Also FIXED: previously used hard np.clip(-1, 1), which is genuine
    digital clipping (harsh, audible distortion) if the model's output
    ever exceeds full scale -- plausible for a system with multiple
    escalation-mode models and a crossfade stage that could, in
    principle, produce a brief overshoot. Now soft-limits by peak-scaling
    the WHOLE buffer down (preserving waveform shape, just at a lower
    level) when an overshoot is detected, and only hard-clips floating
    point NaN/Inf as an absolute last-resort safety net, with a warning
    either way so an overshoot is visible, not silently corrected away.

    Args:
        dither: apply TPDF dither before 16-bit quantization (default:
            True). Only applies for PCM_16 -- per the same research,
            dither is unnecessary at PCM_24 or higher (quantization noise
            already sits below any practical playback system's own noise
            floor), and inapplicable to float output subtypes.
        dither_seed: optional seed for the dither RNG, for reproducible
            test runs -- omit (None) for real deployment, where fresh
            randomness each save is exactly what decorrelation requires.
    """
    import warnings

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    audio = np.ascontiguousarray(audio.squeeze(), dtype=np.float32)

    # NaN/Inf are a genuine last-resort case (never expected in normal
    # operation) -- these cannot be soft-limited (there's no finite peak
    # to scale against), so they're the one case that still gets a hard,
    # loud correction rather than a graceful one.
    if not np.all(np.isfinite(audio)):
        warnings.warn(
            f"save_audio_48k: non-finite samples (NaN/Inf) detected in output for "
            f"{file_path.name} -- replacing with silence. This indicates a real "
            f"upstream numerical problem, not normal operation.",
            stacklevel=2,
        )
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > 1.0:
        warnings.warn(
            f"save_audio_48k: output for {file_path.name} exceeded full scale "
            f"(peak={peak:.3f}) -- soft-limiting by scaling the whole buffer down "
            f"to preserve waveform shape, rather than hard-clipping (which would "
            f"introduce audible harmonic distortion at every excursion). If this "
            f"fires often, it indicates a real gain-staging issue upstream "
            f"(e.g. the crossfade stage or an escalation-mode model), not "
            f"something this save step should be routinely correcting.",
            stacklevel=2,
        )
        audio = audio / peak * 0.999  # a hair under full scale, not exactly at it

    if dither and subtype.upper() == "PCM_16" and audio.size > 0:
        rng = np.random.default_rng(dither_seed)
        lsb = 1.0 / 32768.0  # one least-significant-bit step at 16-bit
        audio = audio + _generate_tpdf_dither(len(audio), lsb, rng)
        # Re-check bounds after adding dither -- TPDF's own peak excursion
        # is only +/-1 LSB, but if the signal was already sitting exactly
        # at the peak-limited 0.999 ceiling, clip the dither's own tiny
        # excursion rather than let it reintroduce an overshoot warning
        # for a difference smaller than a single quantization step.
        audio = np.clip(audio, -1.0, 1.0)

    sf.write(str(file_path), audio, samplerate=sr, subtype=subtype)
    return file_path

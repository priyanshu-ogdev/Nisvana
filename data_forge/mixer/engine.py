"""
Project AEGIS — SNR-Controlled RIR & Noise Mixing Engine
Computes exact acoustic power ratios and convolves spatial room impulse responses.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import scipy.signal as sps


@dataclass
class MixtureResult:
    noisy_audio: np.ndarray
    clean_target: np.ndarray
    reverberant_target: np.ndarray
    target_snr_db: float
    measured_snr_db: float
    speech_power: float
    noise_power: float
    rir_applied: bool


class SnrMixerEngine:
    """Core physics-based acoustic mixing engine."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def convolve_rir(self, speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """
        Convolves speech with Room Impulse Response (RIR) using fast FFT convolution.
        Maintains causality and energy normalization.
        """
        speech = np.asarray(speech, dtype=np.float32)
        rir = np.asarray(rir, dtype=np.float32)

        # Normalize RIR energy to avoid unnatural gain boosts
        rir_energy = np.sum(rir**2)
        if rir_energy > 1e-9:
            rir_norm = rir / np.sqrt(rir_energy)
        else:
            rir_norm = rir

        # Perform fast FFT convolution
        reverb = sps.fftconvolve(speech, rir_norm, mode="full")

        # Trim to original length or keep natural decay tail
        reverb = reverb[: len(speech)]
        return reverb.astype(np.float32)

    def mix_signals(
        self,
        clean_speech: np.ndarray,
        noise: np.ndarray,
        target_snr_db: float,
        rir: Optional[np.ndarray] = None,
    ) -> MixtureResult:
        """
        Mixes clean speech and noise at exact calibrated SNR with optional RIR convolution.
        """
        clean_speech = np.asarray(clean_speech, dtype=np.float32)
        noise = np.asarray(noise, dtype=np.float32)

        # Apply RIR if provided
        rir_applied = rir is not None and len(rir) > 0
        if rir_applied:
            speech_target = self.convolve_rir(clean_speech, rir)
        else:
            speech_target = clean_speech.copy()

        # Match durations: repeat or truncate noise to match speech length
        speech_len = len(speech_target)
        if len(noise) < speech_len:
            repeats = int(np.ceil(speech_len / len(noise)))
            noise = np.tile(noise, repeats)[:speech_len]
        else:
            # Random or starting slice
            noise = noise[:speech_len]

        # Calculate signal powers
        p_speech = float(np.mean(speech_target**2))
        p_noise = float(np.mean(noise**2))

        if p_speech < 1e-12:
            p_speech = 1e-12
        if p_noise < 1e-12:
            p_noise = 1e-12

        # Compute scaling factor: SNR = 10 * log10(P_s / P_n_scaled)
        scale = np.sqrt(p_speech / (p_noise * (10.0 ** (target_snr_db / 10.0))))
        scaled_noise = noise * scale

        # Synthesize noisy mixture
        mixture = speech_target + scaled_noise

        # True-Peak safety limiting: prevent clipping
        peak = np.max(np.abs(mixture))
        if peak > 0.99:
            headroom = 0.99 / peak
            mixture = mixture * headroom
            speech_target = speech_target * headroom
            clean_speech = clean_speech * headroom
            scaled_noise = scaled_noise * headroom

        measured_p_speech = float(np.mean(speech_target**2))
        measured_p_noise = float(np.mean(scaled_noise**2))
        measured_snr = 10.0 * np.log10(max(measured_p_speech, 1e-12) / max(measured_p_noise, 1e-12))

        return MixtureResult(
            noisy_audio=mixture.astype(np.float32),
            clean_target=clean_speech.astype(np.float32),
            reverberant_target=speech_target.astype(np.float32),
            target_snr_db=float(target_snr_db),
            measured_snr_db=round(float(measured_snr), 3),
            speech_power=measured_p_speech,
            noise_power=measured_p_noise,
            rir_applied=rir_applied,
        )

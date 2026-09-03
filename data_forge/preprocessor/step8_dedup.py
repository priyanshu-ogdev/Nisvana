"""
Project AEGIS — Step 8: De-duplication Pass
Computes acoustic and spectral fingerprints across overlapping Freesound derivatives.
"""

import hashlib
from typing import Dict, List, Set, Tuple
import numpy as np


class Deduplicator:
    """Step 8: Acoustic and spectral fingerprint deduplication."""

    def __init__(self, num_subbands: int = 16):
        self.num_subbands = num_subbands
        self.fingerprint_registry: Dict[str, str] = {}  # fingerprint -> clip_id
        self.duplicates: List[Tuple[str, str]] = []     # (duplicate_id, original_id)

    def compute_fingerprint(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        """
        Computes robust acoustic fingerprint based on quantized spectral sub-band energies.
        Invariant to light gain scaling.
        """
        if audio is None or len(audio) == 0:
            return "empty"

        # Downsample or take FFT
        fft_size = 2048
        if len(audio) < fft_size:
            audio = np.pad(audio, (0, fft_size - len(audio)))

        # Average FFT magnitude across time frames
        hop = fft_size // 2
        num_frames = min(len(audio) // hop, 32)
        accum_mag = np.zeros(fft_size // 2 + 1)

        for i in range(num_frames):
            frame = audio[i * hop : i * hop + fft_size] * np.hanning(fft_size)
            mag = np.abs(np.fft.rfft(frame))
            accum_mag += mag

        accum_mag = accum_mag / max(num_frames, 1)

        # Group into log-spaced sub-bands
        bins = np.logspace(0, np.log10(len(accum_mag)), self.num_subbands + 1, dtype=int)
        subband_energies = []
        for b in range(self.num_subbands):
            start, end = bins[b], max(bins[b + 1], bins[b] + 1)
            subband_energies.append(np.mean(accum_mag[start:end]))

        subband_arr = np.array(subband_energies)
        # Normalize to unit sum
        total_e = np.sum(subband_arr) + 1e-12
        norm_subband = subband_arr / total_e

        # Quantize to 8-bit integers
        quantized = (norm_subband * 255.0).astype(np.uint8)

        # Return MD5 hex digest of spectral profile
        return hashlib.md5(quantized.tobytes()).hexdigest()

    def register_and_check(self, clip_id: str, fingerprint: str) -> Tuple[bool, str]:
        """
        Registers fingerprint.
        Returns: (is_duplicate, original_clip_id_if_duplicate)
        """
        if fingerprint in self.fingerprint_registry:
            orig_id = self.fingerprint_registry[fingerprint]
            self.duplicates.append((clip_id, orig_id))
            return True, orig_id

        self.fingerprint_registry[fingerprint] = clip_id
        return False, clip_id

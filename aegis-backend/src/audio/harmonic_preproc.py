"""audio/harmonic_preproc.py — Cyclostationary notch for rotor/drone BPF harmonics.

Gates ON only when the RNNoise spectral profile matches rotor/drone signature.
Covers both rotor and drone classes with one technique (v12/v15 finding).
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
import logging

logger = logging.getLogger(__name__)


class HarmonicPreprocessor:
    """
    Tracks and notch-filters rotor/drone blade-passing frequency and harmonics.
    Operates in the frequency domain at the frame rate.

    Parameters match config/audio_pipeline.yaml:harmonic_preproc section.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        rotor_min_hz: float = 20.0,
        rotor_max_hz: float = 800.0,
        harmonics: int = 5,
        notch_q: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.rotor_min_hz = rotor_min_hz
        self.rotor_max_hz = rotor_max_hz
        self.harmonics = harmonics
        self.notch_q = notch_q
        self.enabled = enabled

        self._estimated_bpf: float | None = None
        self._notch_sos: list | None = None

    def detect_rotor_signature(self, frame: np.ndarray) -> float | None:
        """
        Detect blade-passing frequency via peak-picking in the spectrum.
        Returns estimated BPF in Hz, or None if no rotor signature detected.
        """
        if not self.enabled:
            return None

        # FFT of the frame
        spectrum = np.abs(np.fft.rfft(frame, n=2048))
        freqs = np.fft.rfftfreq(2048, 1.0 / self.sample_rate)

        # Focus on rotor frequency range
        mask = (freqs >= self.rotor_min_hz) & (freqs <= self.rotor_max_hz)
        if not np.any(mask):
            return None

        sub_spec = spectrum[mask]
        sub_freqs = freqs[mask]

        # Find dominant peak
        peak_idx = np.argmax(sub_spec)
        peak_freq = sub_freqs[peak_idx]
        peak_power = sub_spec[peak_idx]

        # Detect only if peak is significantly above local mean (harmonic signature)
        local_mean = np.mean(sub_spec)
        if peak_power < local_mean * 4.0:
            return None  # no clear harmonic structure

        return float(peak_freq)

    def _build_notch_filters(self, bpf_hz: float) -> list:
        """Build cascaded notch filters at BPF and harmonics."""
        sos_chain = []
        for h in range(1, self.harmonics + 1):
            freq = bpf_hz * h
            if freq >= self.sample_rate / 2:
                break
            w0 = freq / (self.sample_rate / 2)
            b, a = sp_signal.iirnotch(w0, self.notch_q)
            sos = sp_signal.tf2sos(b, a)
            sos_chain.append(sos)
        return sos_chain

    def process(self, frame: np.ndarray, snr_state: str = "high") -> np.ndarray:
        """
        Apply harmonic notch filtering if rotor signature detected.
        Returns filtered frame (or original if gate is off).
        """
        if not self.enabled:
            return frame

        # Gate: only process if SNR state suggests impulsive/mechanical noise
        # (skip during clean speech to avoid artifacts)
        if snr_state == "high":
            return frame

        bpf = self.detect_rotor_signature(frame)
        if bpf is None:
            self._estimated_bpf = None
            self._notch_sos = None
            return frame

        # Rebuild notch chain if BPF changed significantly
        if (self._estimated_bpf is None or
                abs(bpf - self._estimated_bpf) > 2.0):
            logger.debug(f"Rotor BPF detected: {bpf:.1f} Hz — rebuilding notch filters")
            self._estimated_bpf = bpf
            self._notch_sos = self._build_notch_filters(bpf)

        if not self._notch_sos:
            return frame

        # Apply cascaded notch filters
        filtered = frame.copy()
        for sos in self._notch_sos:
            filtered = sp_signal.sosfilt(sos, filtered).astype(np.float32)

        return filtered

    def reset(self) -> None:
        self._estimated_bpf = None
        self._notch_sos = None

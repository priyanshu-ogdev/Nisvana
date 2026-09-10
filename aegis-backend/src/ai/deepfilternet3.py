"""ai/deepfilternet3.py — DeepFilterNet3 inference wrapper.

960 FFT / 480 hop @ 48kHz, causal, ~10ms intrinsic delay.
On MacBook dev: falls back to noisereduce when ONNX session is None.
"""
from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DeepFilterNet3:
    def __init__(self, session=None, sample_rate: int = 48000) -> None:
        """session: onnxruntime.InferenceSession or None (triggers noisereduce fallback)."""
        self._session = session
        self.sample_rate = sample_rate
        self._fft_size = 960
        self._hop_size = 480
        self._input_buffer = np.zeros(self._fft_size, dtype=np.float32)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one 10ms frame. Returns enhanced frame."""
        if self._session is not None:
            return self._onnx_infer(frame)
        return self._noisereduce_fallback(frame)

    def _onnx_infer(self, frame: np.ndarray) -> np.ndarray:
        """Real DeepFilterNet3 inference via ONNX Runtime."""
        try:
            # Shift input buffer
            self._input_buffer = np.roll(self._input_buffer, -self._hop_size)
            self._input_buffer[-self._hop_size:] = frame

            # Model expects shape [1, 1, fft_size]
            inp = self._input_buffer.reshape(1, 1, -1)
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: inp})
            enhanced = outputs[0].reshape(-1)[-self._hop_size:]

            # Sanity check — never emit NaN
            if np.any(np.isnan(enhanced)):
                logger.warning("NaN in DeepFilterNet3 output — returning input passthrough")
                return frame

            return enhanced.astype(np.float32)
        except Exception as e:
            logger.error(f"DeepFilterNet3 inference error: {e}")
            return frame

    def _noisereduce_fallback(self, frame: np.ndarray) -> np.ndarray:
        """noisereduce CPU fallback (full frame processing — accumulate 1s then process)."""
        try:
            import noisereduce as nr
            # noisereduce works best on longer clips; for real-time, passthrough with light smoothing
            reduced = nr.reduce_noise(y=frame.astype(np.float64), sr=self.sample_rate, stationary=True)
            return reduced.astype(np.float32)
        except Exception:
            return frame



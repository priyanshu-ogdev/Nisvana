"""ai/cleanumamba.py — CleanUMamba fallback model wrapper (442K params, 8× lighter)."""
from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CleanUMamba:
    """
    Lightweight fallback model for thermal tier 2 (78–82°C).
    On Pi without the ONNX file: degrades gracefully to noisereduce.
    """

    def __init__(self, session=None, sample_rate: int = 48000) -> None:
        self._session = session
        self.sample_rate = sample_rate

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._session is not None:
            try:
                input_name = self._session.get_inputs()[0].name
                inp = frame.reshape(1, 1, -1)
                outputs = self._session.run(None, {input_name: inp})
                enhanced = outputs[0].reshape(-1)
                if np.any(np.isnan(enhanced)):
                    return frame
                return enhanced[-len(frame):].astype(np.float32)
            except Exception as e:
                logger.error(f"CleanUMamba inference error: {e}")
                return frame

        # Fallback: very light spectral subtraction
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=frame.astype(np.float64), sr=self.sample_rate,
                                   stationary=True, prop_decrease=0.5).astype(np.float32)
        except Exception:
            return frame

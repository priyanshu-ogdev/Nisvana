"""audio/codec.py — Audio codec for mesh network transport.

Encodes float32 PCM frames to bytes for WS transmission, decodes back.

Codec selection (auto-detected at import time):
  1. Opus  — if `opuslib` + native `libopus` C library are available.
             ~40-120 bytes per 10ms frame (vs 960 for PCM). Ideal for WAN.
  2. PCM   — int16 fallback. 480 samples × 2 bytes = 960 bytes per 10ms frame.
             Adequate for LAN mesh (5+ nodes over Gigabit).

Phase 1 ships PCM as primary. Opus activates automatically if installed.

Usage:
    codec = AudioCodec()
    payload = codec.encode(pcm_float32_frame)   # → bytes
    frame   = codec.decode(payload)              # → np.ndarray[float32]
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
FRAME_SIZE = 480       # 10ms @ 48kHz
PCM_SCALE = 32767.0    # int16 max


class AudioCodec:
    """
    Bi-directional audio codec for mesh WS transport.

    Automatically selects Opus if available, else falls back to int16 PCM.
    Stateless: each call to encode/decode is independent (no inter-frame state
    in PCM mode; Opus encoder/decoder objects are reused per instance).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._use_opus = False
        self._encoder = None
        self._decoder = None

        self._try_init_opus()

    def _try_init_opus(self) -> None:
        """Attempt to initialise opuslib encoder/decoder. Silently fall back."""
        try:
            import opuslib  # type: ignore
            self._encoder = opuslib.Encoder(
                fs=self.sample_rate,
                channels=1,
                application=opuslib.APPLICATION_VOIP,
            )
            self._decoder = opuslib.Decoder(
                fs=self.sample_rate,
                channels=1,
            )
            self._use_opus = True
            logger.info("AudioCodec: Opus encoder/decoder initialised")
        except Exception as exc:
            logger.info(
                f"AudioCodec: Opus unavailable ({exc!r}). "
                "Using int16 PCM fallback (960 bytes/frame)."
            )
            self._use_opus = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "opus" if self._use_opus else "pcm_int16"

    def encode(self, pcm: np.ndarray) -> bytes:
        """
        Encode a float32 PCM frame to bytes for WS transmission.

        Args:
            pcm: 1D float32 array of length `frame_size`, normalised to [-1, 1].

        Returns:
            Encoded bytes (Opus ~40–120B, PCM exactly 960B for 480 samples).
        """
        pcm_f32 = pcm.astype(np.float32)

        if self._use_opus:
            try:
                # opuslib expects int16 bytes
                pcm_i16 = np.clip(pcm_f32 * PCM_SCALE, -32768, 32767).astype(np.int16)
                return self._encoder.encode(pcm_i16.tobytes(), self.frame_size)
            except Exception as e:
                logger.warning(f"AudioCodec: Opus encode failed ({e}), falling back to PCM")

        # PCM int16 path
        pcm_i16 = np.clip(pcm_f32 * PCM_SCALE, -32768, 32767).astype(np.int16)
        return pcm_i16.tobytes()

    def decode(self, payload: bytes) -> np.ndarray:
        """
        Decode bytes received from the Hub back to a float32 PCM frame.

        Args:
            payload: Bytes as produced by encode().

        Returns:
            1D float32 ndarray of length `frame_size`, normalised to [-1, 1].
        """
        if self._use_opus:
            try:
                raw = self._decoder.decode(payload, self.frame_size)
                pcm_i16 = np.frombuffer(raw, dtype=np.int16)
                return (pcm_i16.astype(np.float32) / PCM_SCALE)
            except Exception as e:
                logger.warning(f"AudioCodec: Opus decode failed ({e}), trying PCM")

        # PCM int16 path
        try:
            pcm_i16 = np.frombuffer(payload, dtype=np.int16)
            if len(pcm_i16) != self.frame_size:
                # Pad or trim to expected frame size
                out = np.zeros(self.frame_size, dtype=np.float32)
                n = min(len(pcm_i16), self.frame_size)
                out[:n] = pcm_i16[:n].astype(np.float32) / PCM_SCALE
                return out
            return pcm_i16.astype(np.float32) / PCM_SCALE
        except Exception as e:
            logger.error(f"AudioCodec: decode failed entirely ({e}), returning silence")
            return np.zeros(self.frame_size, dtype=np.float32)

    def expected_encoded_bytes(self) -> int:
        """Expected size of a PCM-encoded frame in bytes (exact for int16, approx for Opus)."""
        return self.frame_size * 2  # int16 = 2 bytes/sample

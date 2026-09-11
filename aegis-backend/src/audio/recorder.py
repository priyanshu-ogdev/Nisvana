"""audio/recorder.py — Async WAV recorder for downstream mesh audio.

Writes all audio received from the Hub (after decoding) to a local WAV file:
  logs/mesh_audio_<NODE_ID>.wav

Design:
  - Accumulates PCM frames in an in-memory list.
  - Flushes to disk every `flush_interval_frames` frames (~1s at default).
  - Uses `soundfile` on Pi (supports WAV/FLAC/OGG).
  - Falls back to stdlib `wave` on dev machines where soundfile isn't installed.
  - close() performs a final flush and closes the file handle cleanly.

Usage:
    recorder = MeshRecorder("operator-1", log_dir="logs/")
    recorder.write(pcm_frame)     # called from downstream_task
    recorder.flush()              # called periodically
    recorder.close()              # called on shutdown
"""
from __future__ import annotations
import logging
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
FLUSH_INTERVAL_FRAMES = 100   # flush every ~1 second (100 × 10ms frames)


class MeshRecorder:
    """
    Accumulates decoded downstream mesh audio and writes to a WAV file.

    File path: <log_dir>/mesh_audio_<node_id>.wav
    If the file already exists, it is appended to (new session picks up where
    the last one left off). Pass a timestamped node_id to get session files.
    """

    def __init__(
        self,
        node_id: str,
        log_dir: str = "logs/",
        sample_rate: int = SAMPLE_RATE,
        flush_interval_frames: int = FLUSH_INTERVAL_FRAMES,
    ) -> None:
        self.node_id = node_id
        self.sample_rate = sample_rate
        self.flush_interval_frames = flush_interval_frames

        # Ensure log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._wav_path = str(Path(log_dir) / f"mesh_audio_{node_id}.wav")

        self._buffer: List[np.ndarray] = []
        self._frames_since_flush = 0
        self._total_frames_written = 0
        self._file_handle = None   # soundfile.SoundFile or wave.Wave_write
        self._use_soundfile = False

        self._open_file()

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        """Open the WAV file for writing (or appending)."""
        mode = "a" if os.path.exists(self._wav_path) else "w"
        try:
            import soundfile as sf  # type: ignore
            self._file_handle = sf.SoundFile(
                self._wav_path,
                mode=mode,
                samplerate=self.sample_rate,
                channels=1,
                format="WAV",
                subtype="PCM_16",
            )
            self._use_soundfile = True
            logger.info(f"MeshRecorder: opened '{self._wav_path}' (soundfile, mode={mode})")
        except ImportError:
            self._open_stdlib_wave(mode)
        except Exception as e:
            logger.warning(f"MeshRecorder: soundfile open failed ({e}), trying stdlib wave")
            self._open_stdlib_wave(mode)

    def _open_stdlib_wave(self, mode: str) -> None:
        """Fallback: stdlib wave module (write-only, no append — restarts file)."""
        import wave
        try:
            self._file_handle = wave.open(self._wav_path, "w")
            self._file_handle.setnchannels(1)
            self._file_handle.setsampwidth(2)  # int16 = 2 bytes
            self._file_handle.setframerate(self.sample_rate)
            self._use_soundfile = False
            logger.info(f"MeshRecorder: opened '{self._wav_path}' (stdlib wave)")
        except Exception as e:
            logger.error(f"MeshRecorder: could not open WAV file: {e}")
            self._file_handle = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, pcm: np.ndarray) -> None:
        """
        Buffer one frame of float32 PCM for later flush to disk.
        Non-blocking — never waits for I/O.
        """
        self._buffer.append(pcm.astype(np.float32))
        self._frames_since_flush += 1

        if self._frames_since_flush >= self.flush_interval_frames:
            self.flush()

    def flush(self) -> None:
        """Write all buffered frames to disk. Called periodically and on close."""
        if not self._buffer or self._file_handle is None:
            self._buffer.clear()
            self._frames_since_flush = 0
            return

        try:
            audio = np.concatenate(self._buffer, axis=0)
            count = len(self._buffer)

            if self._use_soundfile:
                self._file_handle.write(audio)
                self._file_handle.flush()
            else:
                # stdlib wave: convert float32 → int16
                pcm_i16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
                self._file_handle.writeframes(pcm_i16.tobytes())

            self._total_frames_written += count
            self._buffer.clear()
            self._frames_since_flush = 0
            logger.debug(
                f"MeshRecorder: flushed {count} frames → "
                f"{self._wav_path} (total={self._total_frames_written})"
            )

        except Exception as e:
            logger.error(f"MeshRecorder: flush error: {e}")
            self._buffer.clear()
            self._frames_since_flush = 0

    def close(self) -> None:
        """Final flush and close the WAV file."""
        self.flush()
        if self._file_handle is not None:
            try:
                self._file_handle.close()
                duration_s = self._total_frames_written * 480 / self.sample_rate
                logger.info(
                    f"MeshRecorder: closed '{self._wav_path}' "
                    f"({self._total_frames_written} frames, ~{duration_s:.1f}s)"
                )
            except Exception as e:
                logger.warning(f"MeshRecorder: close error: {e}")
            finally:
                self._file_handle = None

    @property
    def wav_path(self) -> str:
        return self._wav_path

    @property
    def total_frames_written(self) -> int:
        return self._total_frames_written

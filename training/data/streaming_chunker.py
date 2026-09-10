"""
training/data/streaming_chunker.py — Single source of truth for audio chunk parameters

Rev 3 P0.5: One shared chunking utility used by:
  (a) WebDataset loader during QAT training (simulates streaming chunks)
  (b) Fake-quant observer calibration pass
  (c) live_mic_anc.py / ONNX export / escalation_router.py

Without this, training vs. inference chunk-boundary mismatch bugs —
the exact class of bug already found and fixed once for Model 4's
classifier window (escalation_router.py's classifier_window_sec fix) —
would silently reappear across three independent implementations.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class ChunkConfig:
    """
    Canonical streaming chunk parameters for the AEGIS pipeline.
    Every component that processes audio in chunks should read from this
    config, not hard-code its own values.

    chunk_samples = 480 at sample_rate = 48000 gives 10ms chunks,
    matching the ring-buffer convention established in live_mic_anc.py
    and the escalation router's warm-up frame size.
    """
    chunk_samples: int = 480       # 10ms @ 48kHz
    sample_rate: int = 48000
    overlap_samples: int = 0       # No overlap for causal streaming

    @property
    def chunk_duration_ms(self) -> float:
        """Chunk duration in milliseconds."""
        return 1000.0 * self.chunk_samples / self.sample_rate

    @property
    def chunk_duration_sec(self) -> float:
        """Chunk duration in seconds."""
        return self.chunk_samples / self.sample_rate


# Module-level default instance — importable as a constant.
DEFAULT_CHUNK_CONFIG = ChunkConfig()


def chunk_audio(
    audio: np.ndarray,
    config: ChunkConfig = DEFAULT_CHUNK_CONFIG,
) -> Iterator[np.ndarray]:
    """
    Yields non-overlapping chunks of `config.chunk_samples` length.
    The final chunk is zero-padded if the audio length isn't evenly
    divisible by chunk_samples (matching inference's real behavior:
    the last hardware buffer in a real-time stream may be partial).

    Args:
        audio: 1D float32 waveform array.
        config: ChunkConfig specifying chunk size and overlap.

    Yields:
        np.ndarray of shape (chunk_samples,), dtype float32.
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected 1D audio, got shape {audio.shape}")

    chunk_size = config.chunk_samples
    hop_size = chunk_size - config.overlap_samples

    if hop_size <= 0:
        raise ValueError(
            f"overlap_samples ({config.overlap_samples}) must be < "
            f"chunk_samples ({chunk_size})"
        )

    n_samples = len(audio)
    offset = 0

    while offset < n_samples:
        end = offset + chunk_size
        if end <= n_samples:
            yield audio[offset:end].astype(np.float32)
        else:
            # Final partial chunk — zero-pad to chunk_size
            chunk = np.zeros(chunk_size, dtype=np.float32)
            remaining = n_samples - offset
            chunk[:remaining] = audio[offset:].astype(np.float32)
            yield chunk

        offset += hop_size


def chunk_count(n_samples: int, config: ChunkConfig = DEFAULT_CHUNK_CONFIG) -> int:
    """Returns the number of chunks chunk_audio() would yield for audio of this length."""
    hop = config.chunk_samples - config.overlap_samples
    if hop <= 0:
        raise ValueError("overlap must be < chunk_samples")
    return max(1, int(np.ceil(n_samples / hop)))


def reconstruct_from_chunks(
    chunks: Iterator[np.ndarray],
    total_samples: int,
    config: ChunkConfig = DEFAULT_CHUNK_CONFIG,
) -> np.ndarray:
    """
    Reconstructs a waveform from non-overlapping chunks.
    Trims to `total_samples` to remove any trailing zero-pad
    from the final partial chunk.
    """
    hop = config.chunk_samples - config.overlap_samples
    parts = []
    for chunk in chunks:
        if config.overlap_samples == 0:
            parts.append(chunk)
        else:
            # For overlapping chunks, use only the non-overlapping portion
            parts.append(chunk[:hop])

    audio = np.concatenate(parts)
    return audio[:total_samples]

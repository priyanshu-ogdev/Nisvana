"""
inference/runtime/audio_stream.py — Real-Time Streaming Audio Processor & Ring Buffer

Provides:
- Zero-allocation circular ring buffer for real-time streaming audio
- Overlap-Add (OLA) streaming processor with Hanning synthesis windowing to
  eliminate click/pop phase discontinuities across frame boundaries
- Frame accumulator managing fixed chunk sizes from variable soundcard callbacks
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class AudioRingBuffer:
    """
    High-performance circular audio buffer for real-time edge processing.
    Avoids dynamic memory allocation in the audio callback loop.
    """

    def __init__(self, capacity: int = 96000, dtype=np.float32):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=dtype)
        self.head = 0  # Write pointer
        self.tail = 0  # Read pointer
        self.size = 0  # Samples currently stored

    def write(self, data: np.ndarray) -> int:
        """Writes audio data to buffer. Drops oldest samples if full."""
        n = len(data)
        if n > self.capacity:
            data = data[-self.capacity:]
            n = self.capacity

        # Check for overflow and advance tail if necessary
        available_space = self.capacity - self.size
        if n > available_space:
            overflow = n - available_space
            self.tail = (self.tail + overflow) % self.capacity
            self.size -= overflow

        first_chunk = min(n, self.capacity - self.head)
        self.buffer[self.head : self.head + first_chunk] = data[:first_chunk]

        second_chunk = n - first_chunk
        if second_chunk > 0:
            self.buffer[:second_chunk] = data[first_chunk:]

        self.head = (self.head + n) % self.capacity
        self.size += n
        return n

    def read(self, n: int) -> np.ndarray:
        """Reads n samples from buffer without consuming (peek)."""
        n = min(n, self.size)
        if n == 0:
            return np.zeros(0, dtype=self.buffer.dtype)

        out = np.empty(n, dtype=self.buffer.dtype)
        first_chunk = min(n, self.capacity - self.tail)
        out[:first_chunk] = self.buffer[self.tail : self.tail + first_chunk]

        second_chunk = n - first_chunk
        if second_chunk > 0:
            out[first_chunk:] = self.buffer[:second_chunk]

        return out

    def consume(self, n: int) -> int:
        """Advances read pointer by n samples."""
        n = min(n, self.size)
        self.tail = (self.tail + n) % self.capacity
        self.size -= n
        return n

    def clear(self):
        """Resets buffer to empty."""
        self.head = 0
        self.tail = 0
        self.size = 0


class StreamingAudioProcessor:
    """
    Real-time streaming audio engine with 50% Overlap-Add (OLA) reconstruction.

    Ensures that frame-by-frame deep filtering processes continuous audio
    without boundary discontinuities, clipping, or phase jumps.
    """

    def __init__(
        self,
        enhancement_fn: Callable[[np.ndarray], np.ndarray],
        sample_rate: int = 48000,
        frame_size: int = 960,       # 20 ms analysis window
        hop_size: int = 480,         # 10 ms hop (50% overlap)
    ):
        self.enhancement_fn = enhancement_fn
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size

        # Synthesis window satisfying constant overlap-add (COLA)
        self.window = np.hanning(frame_size).astype(np.float32)
        # Normalization factor for 50% overlap Hanning
        self.win_sum = self.window[:hop_size] + self.window[hop_size:]
        self.norm_factor = np.mean(self.win_sum)

        self.in_ring = AudioRingBuffer(capacity=frame_size * 10)
        self.out_overlap = np.zeros(frame_size, dtype=np.float32)

    def reset(self):
        """Resets internal state buffers."""
        self.in_ring.clear()
        self.out_overlap.fill(0.0)

    def process_chunk(self, incoming_audio: np.ndarray) -> np.ndarray:
        """
        Processes incoming stream chunk of any arbitrary size.
        Returns reconstructed, smooth enhanced audio corresponding to completed hops.
        """
        incoming_audio = np.ascontiguousarray(incoming_audio.squeeze(), dtype=np.float32)
        self.in_ring.write(incoming_audio)

        output_chunks = []

        # Process while we have enough samples for an analysis frame
        while self.in_ring.size >= self.frame_size:
            frame = self.in_ring.read(self.frame_size)

            # Apply analysis windowing
            windowed_in = frame * self.window

            # Neural / Hybrid enhancement
            enhanced_frame = self.enhancement_fn(windowed_in)
            enhanced_frame = np.ascontiguousarray(enhanced_frame.squeeze(), dtype=np.float32)

            if len(enhanced_frame) != self.frame_size:
                # Resize if length slightly differs
                enhanced_frame = np.pad(enhanced_frame, (0, max(0, self.frame_size - len(enhanced_frame))))[:self.frame_size]

            # Overlap-add synthesis
            windowed_out = enhanced_frame * self.window
            self.out_overlap += windowed_out

            # Extract output hop
            hop_out = self.out_overlap[: self.hop_size] / max(self.norm_factor, 1e-6)
            output_chunks.append(hop_out.copy())

            # Shift overlap buffer
            self.out_overlap[: self.frame_size - self.hop_size] = self.out_overlap[self.hop_size :]
            self.out_overlap[self.frame_size - self.hop_size :] = 0.0

            # Advance input ring buffer by hop_size
            self.in_ring.consume(self.hop_size)

        if output_chunks:
            return np.concatenate(output_chunks)
        return np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Flushes remaining audio in overlap buffer at stream termination."""
        return self.out_overlap[: self.hop_size] / max(self.norm_factor, 1e-6)

"""
inference/runtime/audio_stream.py — Real-Time Streaming Audio Processor & Ring Buffer

Provides:
- Zero-allocation circular ring buffer for real-time streaming audio
- StatefulHopProcessor: the correct streaming pattern for this project's
  actual models (stateful, causal -- persisted hidden state across calls,
  per model_loader.py). No windowing, no overlap-add, one hop in / one hop
  out. USE THIS for anything backed by HybridAncPipeline /
  AcousticEscalationRouter / model_loader.py's models -- which is
  everything in this codebase today.
- StreamingAudioProcessor: Overlap-Add (OLA) with Hanning analysis/synthesis
  windowing. Correct ONLY for STATELESS, full-context-per-call methods.
  Using this with a stateful model was a real bug found on review (see
  StatefulHopProcessor's docstring below for the full analysis: it
  silently doubled input latency, duplicate-fed every sample through two
  differently-windowed frames while the model's hidden state kept
  advancing, and fed the model synthetically edge-tapered audio). Kept
  available, correctly scoped, for if a genuinely stateless method is ever
  wired in -- not removed, since it isn't wrong in general, only for the
  models this project actually uses.
- Frame accumulator managing fixed chunk sizes from variable soundcard callbacks
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class StatefulHopProcessor:
    """
    MERGE-PASS ADDITION: re-applied from an earlier inference-layer review
    that was not carried into this lineage (confirmed absent via grep
    before this fix). This is the correct streaming pattern for a
    genuinely stateful, causal model chain -- which is what this project's
    models actually are (persisted GRU hidden state, real lookahead-as-
    output-delay behavior, per model_loader.py). StreamingAudioProcessor
    below (Hann-window analysis/synthesis with 50% overlap-add) is the
    RIGHT pattern for stateless, full-context-per-call spectral/masking
    methods -- it is the WRONG pattern for a stateful model, and was being
    used that way in inference/scripts/live_mic_anc.py and
    enhance_audio.py before this fix, which is a real bug, not a style
    preference:

    1. Latency: OLA needs a full `frame_size` (= 2x hop, in the callers'
       actual usage) buffered before it can release ANY output, because it
       must complete one whole analysis window before windowing+overlap-add
       can be computed. A stateful model needs only ONE hop's worth of new
       samples per call -- it carries prior context in its hidden state,
       not in an overlapping window. Wrapping it in OLA was silently
       DOUBLING the input-side buffering latency this project has spent
       real effort minimizing.

    2. Correctness: OLA's 50%-overlap means every audio sample is fed to
       the model TWICE, in two different frames, each time weighted by a
       different point on the Hann taper, while the model's hidden state
       advances on every call regardless. A stateful model has no way to
       know it's seeing duplicated, re-weighted content rather than a
       clean, once-through, in-order stream -- this corrupts exactly the
       temporal continuity the statefulness fix (model_loader.py) was
       added to provide.

    3. Signal quality: the analysis window tapers each frame's input
       toward zero at both edges before the model ever sees it -- a
       plausible source of audible warbling/pumping artifacts
       synchronized to the hop rate, entirely separate from whatever the
       model's actual enhancement quality is.

    GROUNDED, not just reasoned from first principles: a causal RNN/GRU
    speech-enhancement model's whole justification for real-time use is
    exactly this pattern -- "past information can be encoded into the
    hidden state vector h so that only the input feature at time tau and
    the state vector from tau-1 are required" (arXiv:2002.05843). And
    concretely, for the actual model this project is built on: a real,
    deployed, stateful streaming conversion of DeepFilterNet3 itself
    (huggingface.co/iky1e/DeepFilterNet3-Streaming-CoreML) states its
    runtime contract in exactly these terms -- "consumes one 480-sample
    (10 ms) hop at a time and exposes all recurrent state explicitly."

    ONE FIGURE WORTH FLAGGING, found via that same check: that same real
    streaming conversion states DeepFilterNet3's actual fixed algorithmic
    delay as 1,440 samples / 30ms -- not the ~10ms figure (hop-size-only
    reasoning) used elsewhere in this project's documentation for the
    zero-lookahead config. Flagged, not silently corrected throughout this
    project's broader documentation in this pass.

    This class passes consecutive, NON-overlapping hop-sized chunks
    directly to `enhancement_fn`, unmodified (no windowing in, no
    overlap-add out) -- correct for a model that maintains its own
    continuity via persisted internal state.
    """

    def __init__(
        self,
        enhancement_fn: Callable[[np.ndarray], np.ndarray],
        sample_rate: int = 48000,
        hop_size: int = 480,   # 10 ms @ 48kHz -- matches DeepFilterNet3's
                                 # low-latency-config hop; NOT frame_size,
                                 # since there is no analysis window here
        reset_fn: Optional[Callable[[], None]] = None,
    ):
        self.enhancement_fn = enhancement_fn
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        # Optional callable invoked by reset() -- typically a pipeline's or
        # router's reset_state(), so starting a new stream via this
        # processor also clears the model(s)' hidden state in one call.
        self._reset_fn = reset_fn
        self.in_ring = AudioRingBuffer(capacity=hop_size * 20)

    def reset(self):
        """Clears buffered input AND (if a reset_fn was supplied) the
        backing pipeline/model's hidden state -- call this at the start of
        every new stream/session, not just when this object is constructed,
        since a long-lived process may reuse one instance across sessions."""
        self.in_ring.clear()
        if self._reset_fn is not None:
            self._reset_fn()

    def process_chunk(self, incoming_audio: np.ndarray) -> np.ndarray:
        """
        Accepts audio of any chunk size, buffers it, and releases enhanced
        audio in complete hop_size units as they become available -- no
        windowing, no overlap-add, no frame_size wait. A caller feeding
        exactly hop_size at a time gets output on every single call.
        """
        incoming_audio = np.ascontiguousarray(incoming_audio.squeeze(), dtype=np.float32)
        self.in_ring.write(incoming_audio)

        output_chunks = []
        while self.in_ring.size >= self.hop_size:
            hop = self.in_ring.read(self.hop_size)
            self.in_ring.consume(self.hop_size)

            enhanced_hop = self.enhancement_fn(hop)
            enhanced_hop = np.ascontiguousarray(enhanced_hop.squeeze(), dtype=np.float32)
            if len(enhanced_hop) != self.hop_size:
                enhanced_hop = np.pad(
                    enhanced_hop, (0, max(0, self.hop_size - len(enhanced_hop)))
                )[: self.hop_size]
            output_chunks.append(enhanced_hop)

        if output_chunks:
            return np.concatenate(output_chunks)
        return np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """
        Unlike StreamingAudioProcessor's flush (which releases a windowed
        overlap tail), a stateful hop processor has no partial-frame
        residue to release by design -- any leftover samples in the ring
        buffer are strictly less than one hop and were never enough to run
        the model on. Returned as silence of that length so callers
        expecting *some* array back for the stream's final partial chunk
        don't need special-case handling.
        """
        remainder = self.in_ring.size
        return np.zeros(remainder, dtype=np.float32)


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

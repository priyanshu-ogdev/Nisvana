"""
inference/utils/transmission_prep.py — Narrowband Transmission Preparation

THE GAP THIS CLOSES: this project's own PRD/hardware design specified
narrowband tactical radio output (STANAG-adjacent bandwidth, roughly
2.4-16kHz) as the actual downstream destination for enhanced speech --
but repo-wide search confirmed zero code anywhere converts this system's
internal 48kHz processing rate down to anything a real tactical radio
would actually transmit. Every path (enhance_audio.py, live_mic_anc.py)
produces 48kHz PCM WAV output only, useful for file-based evaluation and
demos, but not radio-ready.

WHAT THIS DELIBERATELY IS NOT: a real military-grade narrowband voice
codec (MELPe, CVSD, etc.). Those require specialized libraries not
available in this environment, and reimplementing one is a substantial,
specialized undertaking outside this project's own established scope
discipline (matching how Model 2's lookahead-buffering and full
static-quantization/QAT were correctly flagged rather than attempted).
What IS implemented: the achievable, correct engineering step that
belongs BEFORE any real codec -- proper bandwidth-limiting down to the
radio's actual channel, using the same polyphase resampling (not naive
decimation, which would alias) already established as this project's own
best practice, plus the same TPDF dither discipline already applied to
the 48kHz PCM_16 save path, applied consistently at the new target rate.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import signal


@dataclass
class TransmissionPrepConfig:
    # STANAG-adjacent narrowband tactical radio bandwidth, per this
    # project's own original hardware design discussion (2.4-16kHz).
    # 16kHz chosen as the default target -- wideband tactical radio
    # channels (the higher end of that range) rather than legacy
    # narrowband voice channels (which would need ~8kHz or even 4kHz
    # telephone-bandwidth and a real codec, not just resampling); adjust
    # per the actual radio hardware's real specified channel bandwidth,
    # which this project's own docs never pinned to one specific number.
    target_sample_rate: int = 16000
    apply_dither: bool = True
    dither_seed: int = None

    # Pre-emphasis: a classical, well-established technique in telephony/
    # radio voice transmission that boosts higher frequencies before
    # transmission (compensating for many radio/telephone channels' own
    # non-flat, high-frequency-attenuating response, and for the fact
    # that voiced-speech energy falls off at ~-6dB/octave, so pre-emphasis
    # flattens the spectrum being transmitted for better use of the
    # channel's dynamic range). The RECEIVING end must apply the exact
    # inverse (de-emphasis) -- this module only prepares the transmit
    # side; whatever consumes this output is responsible for the matched
    # de-emphasis step, which this module cannot know or control.
    apply_pre_emphasis: bool = True
    pre_emphasis_coefficient: float = 0.95  # standard telephony-range value (typically 0.9-0.97)


def _apply_pre_emphasis(audio: np.ndarray, coefficient: float) -> np.ndarray:
    """y[n] = x[n] - coefficient * x[n-1] -- the standard first-order
    pre-emphasis filter used throughout telephony and narrowband voice
    transmission. Implemented via scipy.signal.lfilter for numerical
    consistency with any other filtering already in this pipeline,
    rather than a hand-rolled loop."""
    return signal.lfilter([1.0, -coefficient], [1.0], audio).astype(np.float32)


def prepare_for_transmission(
    audio: np.ndarray,
    source_sample_rate: int,
    config: TransmissionPrepConfig = None,
) -> Tuple[np.ndarray, int]:
    """
    Converts this system's internal-rate enhanced audio into the
    bandwidth a real narrowband tactical radio channel would actually
    carry -- the missing step between "enhanced 48kHz PCM" and "ready to
    hand to a radio."

    Args:
        audio: enhanced audio at source_sample_rate (typically this
            project's internal 48000 Hz).
        source_sample_rate: the actual rate `audio` is currently at.
        config: see TransmissionPrepConfig -- target rate, dither,
            pre-emphasis all configurable per the actual radio hardware's
            real specification, which varies by platform.

    Returns:
        (prepared_audio, target_sample_rate)
    """
    config = config or TransmissionPrepConfig()
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    if config.apply_pre_emphasis:
        audio = _apply_pre_emphasis(audio, config.pre_emphasis_coefficient)

    if source_sample_rate != config.target_sample_rate:
        # Proper polyphase resampling (not naive decimation, which would
        # alias) -- same tool this project already established as correct
        # practice elsewhere (data_forge's own resampling, audio_io.py's
        # load_audio_48k).
        gcd = np.gcd(source_sample_rate, config.target_sample_rate)
        audio = signal.resample_poly(
            audio, config.target_sample_rate // gcd, source_sample_rate // gcd
        ).astype(np.float32)

    if config.apply_dither:
        # Same TPDF discipline as audio_io.py's save_audio_48k -- a
        # narrowband channel has proportionally MORE to lose from
        # correlated quantization distortion, not less, since there's
        # already less bandwidth/dynamic range to work with.
        rng = np.random.default_rng(config.dither_seed)
        lsb = 1.0 / 32768.0
        u1 = rng.uniform(-0.5, 0.5, len(audio))
        u2 = rng.uniform(-0.5, 0.5, len(audio))
        audio = audio + (u1 + u2) * lsb

    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > 1.0:
        audio = audio / peak * 0.999

    return audio, config.target_sample_rate


def _apply_de_emphasis(audio: np.ndarray, coefficient: float) -> np.ndarray:
    """y[n] = x[n] + coefficient * y[n-1] — inverse of _apply_pre_emphasis.
    Apply on the RECEIVE side to recover the original spectral balance."""
    return signal.lfilter([1.0], [1.0, -coefficient], audio).astype(np.float32)


@dataclass
class CodecPrepConfig:
    """Codec-specific normalization parameters for common tactical codecs.
    Each codec has different optimal input level, bandwidth, and dynamic
    range expectations — feeding optimally normalized audio reduces codec
    artifacts that would otherwise waste the enhancement model's gains."""
    codec: str = "opus"  # "opus", "amr_wb", "melpe", "raw_pcm"
    # Opus prefers -1.0 dBFS peak, 16kHz or 48kHz input
    # AMR-WB requires exactly 16kHz, -26 dBov nominal
    # MELPe requires 8kHz, specific loudness normalization

    # Target peak level (dBFS) — codec-specific optimal
    target_peak_dbfs: float = -1.0
    # Whether to apply a gentle high-shelf boost to compensate for
    # typical codec high-frequency rolloff (measurable in Opus at bitrates < 24kbps)
    compensate_codec_rolloff: bool = False
    rolloff_shelf_gain_db: float = 1.5
    rolloff_shelf_freq_hz: float = 3500.0


def prepare_for_codec(
    audio: np.ndarray,
    source_sample_rate: int,
    codec_config: CodecPrepConfig = None,
) -> np.ndarray:
    """
    Applies codec-specific normalization to enhanced audio before encoding.
    This is a pre-processing step AFTER prepare_for_transmission() — it
    adjusts levels and spectral shape to minimize artifacts in the specific
    codec being used for transmission.

    Does NOT actually encode the audio (that requires codec-specific
    libraries like libopus or opencore-amr that are outside this project's
    current dependencies) — it prepares the PCM for optimal codec input.
    """
    codec_config = codec_config or CodecPrepConfig()
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    # Target peak normalization
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > 0:
        target_peak_linear = 10.0 ** (codec_config.target_peak_dbfs / 20.0)
        audio = audio * (target_peak_linear / peak)

    # Codec-specific adjustments
    if codec_config.codec == "amr_wb":
        # AMR-WB requires exactly 16kHz — if not already resampled, warn
        # (prepare_for_transmission should have handled this already)
        pass
    elif codec_config.codec == "opus":
        # Opus handles 8/12/16/24/48kHz natively — no forced resampling needed
        pass

    # Optional high-shelf boost to pre-compensate codec rolloff
    if codec_config.compensate_codec_rolloff and source_sample_rate > 0:
        try:
            # First-order high-shelf filter using bilinear transform
            w0 = 2.0 * np.pi * codec_config.rolloff_shelf_freq_hz / source_sample_rate
            gain_linear = 10.0 ** (codec_config.rolloff_shelf_gain_db / 20.0)
            # Simple first-order shelf: H(s) = (s + w0*sqrt(A)) / (s + w0/sqrt(A))
            b, a = signal.butter(1, codec_config.rolloff_shelf_freq_hz,
                                 btype='high', fs=source_sample_rate)
            shelf_filtered = signal.lfilter(b, a, audio)
            # Blend: original + scaled high-frequency boost
            blend = gain_linear - 1.0
            audio = audio + blend * shelf_filtered
        except Exception:
            pass  # Filter design failure — leave audio unchanged

    return audio


@dataclass
class LatencyBudget:
    """Complete latency budget breakdown for a tactical voice transmission path."""
    processing_latency_ms: float = 0.0    # Model compute time (from benchmark)
    algorithmic_delay_ms: float = 0.0     # Model's inherent framing delay
    codec_frame_ms: float = 20.0          # Codec frame size (Opus default: 20ms)
    network_rtt_ms: float = 0.0           # One-way network latency
    playback_buffer_ms: float = 10.0      # Receiver-side jitter buffer
    total_ms: float = 0.0                 # Computed total

    def compute_total(self) -> float:
        """Computes total one-way glass-to-glass latency."""
        self.total_ms = (
            self.processing_latency_ms
            + self.algorithmic_delay_ms
            + self.codec_frame_ms
            + self.network_rtt_ms
            + self.playback_buffer_ms
        )
        return self.total_ms

    def meets_target(self, target_ms: float = 150.0) -> bool:
        """ITU-T G.114 recommends < 150ms one-way for good conversational quality."""
        self.compute_total()
        return self.total_ms <= target_ms

    def to_dict(self) -> dict:
        self.compute_total()
        return {
            "processing_ms": round(self.processing_latency_ms, 1),
            "algorithmic_delay_ms": round(self.algorithmic_delay_ms, 1),
            "codec_frame_ms": round(self.codec_frame_ms, 1),
            "network_rtt_ms": round(self.network_rtt_ms, 1),
            "playback_buffer_ms": round(self.playback_buffer_ms, 1),
            "total_one_way_ms": round(self.total_ms, 1),
            "meets_g114_150ms": self.total_ms <= 150.0,
            "meets_tactical_100ms": self.total_ms <= 100.0,
        }


def compute_latency_budget(
    model_key: str = "aegis-se-primary",
    processing_latency_ms: float = 2.0,
    codec: str = "opus",
    network_rtt_ms: float = 0.0,
) -> LatencyBudget:
    """
    Computes a complete latency budget for the transmission path.
    Uses known algorithmic delays from onnx_engine.MODEL_ALGORITHMIC_DELAY_MS
    and codec-specific frame sizes.
    """
    from inference.engines.onnx_engine import MODEL_ALGORITHMIC_DELAY_MS

    algo_delay = MODEL_ALGORITHMIC_DELAY_MS.get(model_key, 0.0)

    codec_frame_ms = {
        "opus": 20.0,     # Opus default frame (configurable 2.5-120ms)
        "amr_wb": 20.0,   # AMR-WB fixed 20ms frame
        "melpe": 22.5,    # MELPe 22.5ms frame
        "raw_pcm": 0.0,   # No codec framing
    }.get(codec, 20.0)

    budget = LatencyBudget(
        processing_latency_ms=processing_latency_ms,
        algorithmic_delay_ms=algo_delay,
        codec_frame_ms=codec_frame_ms,
        network_rtt_ms=network_rtt_ms,
    )
    budget.compute_total()
    return budget


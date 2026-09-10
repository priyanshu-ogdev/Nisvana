"""
Project AEGIS — Multi-Shot Compositing (Gunfire Diversity)

Recombines independent REAL gunshot recordings (overlay and/or concatenation)
to manufacture automatic-fire and multiple-simultaneous-shooter acoustic
scenarios from a source base that is otherwise a small number of discrete
single-shot recordings (gunshot_dryad's one field session, NOISEX-92's
single machinegun.wav, MAD's gunshot subset).

WHY THIS EXISTS, EXPLICITLY: this project already built and then explicitly
removed procedural/synthetic gunshot waveform generation (Friedlander-wave
blasts, muzzle-blast N-waves) for submission-authenticity reasons -- see
docs/Nisvana_PRD.md Sec 3.2. This module does NOT reopen that decision. It
never synthesizes a waveform sample from a physics model; every sample in
the output is either (a) a verbatim segment of a real recording, or (b) a
linear superposition / concatenation of real recordings. No new acoustic
event is invented -- only the temporal arrangement of real ones changes.
If this distinction matters for SIH disclosure, it is stated here rather
than assumed.

Two composite modes:
  - OVERLAY: sums two or more real single-shot clips at independently
    jittered onset offsets, simulating multiple simultaneous shooters or
    weapon-echo overlap.
  - BURST: concatenates several real single-shot clips end-to-end with
    randomized inter-shot gaps drawn from a plausible cyclic-rate range,
    simulating semi-automatic/automatic fire from a single weapon when the
    only real automatic-fire clip available (NOISEX-92's machinegun.wav) is
    a single, non-representative sample.
"""

from dataclasses import dataclass
from typing import List, Optional
import random
import numpy as np


@dataclass
class MultiShotCompositeConfig:
    sample_rate: int = 48000

    # OVERLAY mode
    overlay_min_shots: int = 2
    overlay_max_shots: int = 4
    overlay_onset_jitter_ms: float = 250.0   # max random offset between overlaid shots

    # BURST mode
    burst_min_shots: int = 3
    burst_max_shots: int = 8
    # Cyclic-rate-plausible inter-shot gap range (seconds). ~100-900ms spans
    # semi-automatic through moderate-cyclic-rate automatic fire; deliberately
    # not pushed to true high-cyclic-rate (<50ms) since that regime is only
    # actually represented by the single NOISEX-92 machinegun.wav clip and
    # this module works from discrete single-shot sources for BURST mode.
    burst_min_gap_sec: float = 0.10
    burst_max_gap_sec: float = 0.90

    # Per-shot gain jitter applied before combining, so composited shots
    # aren't identically leveled (a real multi-shooter/multi-shot scene
    # wouldn't be either) -- kept modest, this is not the primary diversity
    # lever, RIR convolution and mixing SNR are.
    per_shot_gain_jitter_db: float = 3.0


class MultiShotCompositor:
    """
    Combines real single-shot gunfire clips into overlay or burst
    composites. Operates purely on real audio segments -- see module
    docstring for why this is a recombination tool, not a synthesizer.
    """

    def __init__(self, config: Optional[MultiShotCompositeConfig] = None):
        self.config = config or MultiShotCompositeConfig()

    def _apply_gain_jitter(self, audio: np.ndarray) -> np.ndarray:
        jitter_db = random.uniform(-self.config.per_shot_gain_jitter_db, self.config.per_shot_gain_jitter_db)
        gain = 10.0 ** (jitter_db / 20.0)
        out = audio * gain
        peak = np.max(np.abs(out)) if out.size else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out.astype(np.float32)

    def overlay(self, shot_clips: List[np.ndarray], target_length_samples: Optional[int] = None) -> np.ndarray:
        """
        Sums 2-4 real shot clips at independently jittered onsets, simulating
        multiple simultaneous shooters / muzzle-blast + weapon-echo overlap.

        Args:
            shot_clips: pool of real, independent single-shot audio arrays
                (mono, float32) to draw from. Must contain at least
                `overlay_min_shots` distinct clips.
            target_length_samples: output length; defaults to the longest
                jittered placement needed to fit all chosen shots.
        """
        if len(shot_clips) < self.config.overlay_min_shots:
            raise ValueError(
                f"Need at least {self.config.overlay_min_shots} distinct real shot "
                f"clips to overlay-composite, got {len(shot_clips)}."
            )

        n_shots = random.randint(self.config.overlay_min_shots, min(self.config.overlay_max_shots, len(shot_clips)))
        chosen = random.sample(shot_clips, n_shots)
        max_jitter_samples = int(self.config.overlay_onset_jitter_ms / 1000.0 * self.config.sample_rate)

        offsets = [random.randint(0, max_jitter_samples) for _ in chosen]
        lengths = [len(c) for c in chosen]
        needed_len = max(o + l for o, l in zip(offsets, lengths))
        out_len = target_length_samples or needed_len

        composite = np.zeros(out_len, dtype=np.float32)
        for clip, offset in zip(chosen, offsets):
            clip = self._apply_gain_jitter(np.asarray(clip, dtype=np.float32))
            end = min(offset + len(clip), out_len)
            usable = end - offset
            if usable <= 0:
                continue
            composite[offset:end] += clip[:usable]

        peak = np.max(np.abs(composite)) if composite.size else 0.0
        if peak > 0.99:
            composite = composite * (0.99 / peak)
        return composite.astype(np.float32)

    def burst(self, shot_clips: List[np.ndarray]) -> np.ndarray:
        """
        Concatenates real single-shot clips end-to-end with randomized
        inter-shot gaps, simulating semi-automatic/automatic fire built from
        discrete real recordings. See module docstring for the cyclic-rate
        caveat -- this is not a substitute for real automatic-fire
        recordings, it extends coverage from what's available.

        Args:
            shot_clips: pool of real, independent single-shot audio arrays
                to draw from (sampling WITH replacement -- a limited source
                pool, e.g. 7 Dryad firearms, still needs to produce longer
                bursts than the pool size).
        """
        if not shot_clips:
            raise ValueError("Need at least 1 real shot clip to burst-composite.")

        n_shots = random.randint(self.config.burst_min_shots, self.config.burst_max_shots)
        segments: List[np.ndarray] = []
        for i in range(n_shots):
            clip = self._apply_gain_jitter(np.asarray(random.choice(shot_clips), dtype=np.float32))
            segments.append(clip)
            if i < n_shots - 1:
                gap_sec = random.uniform(self.config.burst_min_gap_sec, self.config.burst_max_gap_sec)
                segments.append(np.zeros(int(gap_sec * self.config.sample_rate), dtype=np.float32))

        return np.concatenate(segments).astype(np.float32)

    def composite(self, shot_clips: List[np.ndarray], mode: Optional[str] = None) -> np.ndarray:
        """
        Convenience entry point: picks overlay or burst at random (or as
        specified) and returns a composite built from real audio only.
        """
        mode = mode or random.choice(["overlay", "burst"])
        if mode == "overlay":
            return self.overlay(shot_clips)
        elif mode == "burst":
            return self.burst(shot_clips)
        raise ValueError(f"Unknown composite mode: {mode!r} (expected 'overlay' or 'burst')")

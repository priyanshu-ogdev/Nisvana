"""
inference/runtime/multichannel_frontend.py — Hardware Input Bridge

THE GAP THIS CLOSES: this project's original hardware design (established
early in this project, before the pivot to the data-forge/DeepFilterNet3/
CleanUMamba architecture) specified a multi-microphone array plus a
throat-contact microphone as the physical input -- spatial beamforming as
the first processing tier, ahead of any AI enhancement. Repo-wide search
confirmed zero trace of that design anywhere in the current inference or
model layer: every model in model_loader.py takes single-channel (mono)
input. That's not a bug in any one file -- it's a design requirement that
was never carried forward when the project pivoted to real, buildable
open-source single-channel backbones under this project's own established
time constraints, and never explicitly re-stated as a scope decision
either. This module is the honest bridge: real multi-mic + throat-mic
hardware input in, single mono stream out, feeding the existing pipeline
unchanged.

WHAT THIS DELIBERATELY IS NOT: a learned or calibrated beamformer (MVDR/
GSC/neural). Building and training one was explicitly out of scope given
this project's repeatedly-stated time constraints, and doing it as an
afterthought bolted onto an already-built single-channel pipeline would
be worse than being honest about a classical, zero-calibration default.
The two operations below are deliberately simple and dependency-light:

1. Energy-weighted air-mic combination -- NOT a geometry-aware delay-and-
   sum (that requires known mic spacing/geometry, which was never fixed
   in this project's hardware spec), but a robust, calibration-free
   default: channels with more signal energy (closer to the speaker, or
   less occluded) are weighted more heavily than uniform averaging would.
2. Throat-mic energy-gated blending -- throat/bone-conduction mics are
   physically near-immune to airborne acoustic noise (they pick up
   vocal-tract vibration directly), at the cost of poor high-frequency
   consonant fidelity. The classical, well-established use of a throat
   mic alongside an air-mic array is exactly this: lean on it more when
   the air array's estimated SNR is poor, lean on the air array more
   otherwise. This is implemented as a simple SNR-gated linear blend,
   not a learned fusion network, for the same "buildable now, not a
   research project" reason as (1).

UPGRADE PATH, stated explicitly rather than left implicit: if the mic
array's physical geometry (spacing, arrangement) is ever fixed in the
hardware spec, replace `_combine_air_mics` with a real delay-and-sum or
MVDR beamformer (`pyroomacoustics` was the specific tool already
identified for this in this project's own earlier design work) --
the rest of this module's interface (N-channel array + throat-mic in,
mono out) does not need to change for that upgrade to happen.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HardwareFrontendConfig:
    num_air_mics: int = 4                  # matches this project's original array spec
    has_throat_mic: bool = True
    sample_rate: int = 48000

    # Energy-weighting smoothing -- avoids the per-sample weight jittering
    # wildly on a single loud transient (a gunshot on one mic shouldn't
    # instantly zero out the others for the rest of the utterance).
    energy_window_ms: float = 20.0

    # Throat-mic blend gating. estimated_snr_db below this threshold shifts
    # weight toward the throat mic; above it, the air array dominates.
    # Reasoned default, NOT a literature citation -- no published guidance
    # for this exact gating threshold was found; treat as a starting point
    # to validate against real hardware recordings, same as every other
    # reasoned-default parameter elsewhere in this design.
    throat_mic_gate_snr_db: float = 5.0
    throat_mic_max_weight: float = 0.6     # even at its most trusted, the air array still
                                            # contributes -- throat mic alone loses too much
                                            # high-frequency consonant content to go it alone


class MultichannelHardwareFrontend:
    """
    Converts real hardware input (N air-mic channels + optional throat-mic
    channel) into the single mono stream every downstream model
    (aegis-se-primary, aegis-se-escalation, aegis-se-crosscheck,
    aegis-clf-gate) actually consumes.

    Call `process(air_mic_channels, throat_mic_channel, estimated_snr_db)`
    once per audio chunk, ahead of the AcousticEscalationRouter.
    """

    def __init__(self, config: Optional[HardwareFrontendConfig] = None):
        self.config = config or HardwareFrontendConfig()
        self._energy_smooth_state: Optional[np.ndarray] = None

    def reset_state(self) -> None:
        self._energy_smooth_state = None

    def _combine_air_mics(self, air_mic_channels: np.ndarray) -> np.ndarray:
        """
        air_mic_channels: shape (num_air_mics, chunk_samples).
        Returns: shape (chunk_samples,) -- energy-weighted combination.
        """
        if air_mic_channels.shape[0] == 1:
            return air_mic_channels[0]

        # Per-channel RMS energy over this chunk -- the calibration-free
        # weighting signal described in the module docstring.
        per_channel_energy = np.sqrt(np.mean(air_mic_channels ** 2, axis=1) + 1e-12)

        # Smooth across chunks so one transient on one mic doesn't cause a
        # full weight swing on the very next chunk.
        if self._energy_smooth_state is None or self._energy_smooth_state.shape != per_channel_energy.shape:
            self._energy_smooth_state = per_channel_energy
        else:
            alpha = 0.3  # matches this module's own "reasoned default, not cited" standard
            self._energy_smooth_state = alpha * per_channel_energy + (1 - alpha) * self._energy_smooth_state

        weights = self._energy_smooth_state / (np.sum(self._energy_smooth_state) + 1e-12)
        combined = np.tensordot(weights, air_mic_channels, axes=(0, 0))
        return combined

    def _blend_throat_mic(
        self,
        air_combined: np.ndarray,
        throat_mic_channel: np.ndarray,
        estimated_snr_db: float,
    ) -> np.ndarray:
        cfg = self.config
        if estimated_snr_db >= cfg.throat_mic_gate_snr_db:
            throat_weight = 0.0
        else:
            # Linear ramp from 0 at the gate threshold up to throat_mic_max_weight
            # as SNR drops further -- saturates at -20dB below the gate so a
            # single extreme transient doesn't produce a divide-by-huge-number
            # style discontinuity.
            deficit = cfg.throat_mic_gate_snr_db - estimated_snr_db
            throat_weight = min(cfg.throat_mic_max_weight, cfg.throat_mic_max_weight * (deficit / 20.0))

        return (1.0 - throat_weight) * air_combined + throat_weight * throat_mic_channel

    def _estimate_local_snr_proxy(self, air_combined: np.ndarray) -> float:
        """
        Cheap, local SNR proxy so this module doesn't need to wait on
        AcousticEscalationRouter.analyze_audio's own estimate (which is
        normally computed on THIS module's output -- calling it first
        would be circular). Deliberately the same crest-factor heuristic
        already used there, not a second independent method, so the two
        components' SNR notions stay consistent with each other rather
        than silently disagreeing.
        """
        rms = float(np.sqrt(np.mean(air_combined ** 2)) + 1e-8)
        peak = float(np.max(np.abs(air_combined)))
        crest = peak / rms
        return float(10.0 * np.log10(max(crest, 1.0)) * 2.0 - 5.0)

    def process(
        self,
        air_mic_channels: np.ndarray,
        throat_mic_channel: Optional[np.ndarray] = None,
        estimated_snr_db: Optional[float] = None,
    ) -> np.ndarray:
        """
        air_mic_channels: shape (num_air_mics, chunk_samples) -- real
            hardware capture, num_air_mics matching config.num_air_mics.
        throat_mic_channel: shape (chunk_samples,), or None if the hardware
            variant in use doesn't have one (config.has_throat_mic=False).
        estimated_snr_db: optional external estimate (e.g. from a caller
            that already has one). If None, computed locally via
            `_estimate_local_snr_proxy` -- this module does NOT depend on
            AcousticEscalationRouter.analyze_audio running first, since
            that would be circular (it normally runs on this module's
            OUTPUT).

        Returns: shape (chunk_samples,) mono stream, ready for
            AcousticEscalationRouter.route_and_enhance.
        """
        if air_mic_channels.ndim == 1:
            air_mic_channels = air_mic_channels[np.newaxis, :]

        air_combined = self._combine_air_mics(air_mic_channels)

        if not self.config.has_throat_mic or throat_mic_channel is None:
            return air_combined

        if estimated_snr_db is None:
            estimated_snr_db = self._estimate_local_snr_proxy(air_combined)

        return self._blend_throat_mic(air_combined, throat_mic_channel, estimated_snr_db)

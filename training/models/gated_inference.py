"""
training/models/gated_inference.py — Confidence-gated enhancement wrapper.

Closes design gap #2 from the AEGIS ML-layer generalization review: the
Model 4 classifier's speech_dominant output previously only drove WHICH SE
model (1 vs 2) handles a chunk (escalation routing, per model_loader.py's
docstring) -- it never touched HOW AGGRESSIVELY either model suppresses.
A chunk confidently classified as speech_dominant is direct evidence
"this isn't noise, don't suppress it," but nothing used that evidence at
the enhancement-OUTPUT stage, only at checkpoint-selection time
(WorstClassCheckpointSelector's PASSTHROUGH_PROTECTED_CLASSES guard,
added in the previous review pass -- which protects TRAINING from
learning to over-suppress clean speech, but can't protect a single
inference call from occasionally over-suppressing regardless of how
well-trained the model already is).

DESIGN: a confidence-gated dry/wet blend, not a retrained joint
architecture. Reasoning, stated plainly since this is a project-specific
engineering choice, not a literature citation:

  - Zero additional training cost, and no risk of a jointly-trained model
    landing in a worse local optimum than either the standalone SE
    trainers or the standalone classifier trainer already validate
    independently.
  - The classifier (Model 4) and the SE models (Models 1-3) already have
    different training schedules, loss functions, and update cadences in
    this codebase (see train_pipeline.py's stage ordering) -- a blend at
    the wrapper boundary lets either be retrained/improved without
    forcing a joint re-tune of the other.
  - This mirrors the same "gate an enhancement stage by an auxiliary
    confidence signal" pattern already used elsewhere in this project's
    architecture (the harmonic/SNR-state gating ahead of AEC in the
    backbone design), not a new paradigm introduced here.

SMOOTHING: the gate value is exponentially smoothed across chunks
(attack/release, the same idea as an audio compressor's gain-reduction
smoothing) specifically to avoid audible "pumping" -- an un-smoothed gate
could otherwise flip abruptly between adjacent enhancement chunks (SE
models stream at ~10ms) and classifier windows (trained on 0.2s windows,
per classifier_trainer.py's own log line), which is a much coarser
update rate than the audio itself.

NOT a bypass switch: `min_enhancement_strength` floors how far the gate
can relax suppression, because background noise underneath speech (e.g.
a vehicle idling behind a radio call) still needs some suppression even
when the classifier is fully confident the CURRENT window is
speech-dominant.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GatedEnhancementConfig:
    # Classifier softmax index for "speech_dominant" -- must match the
    # class order classifier_trainer.py trains against:
    # [harmonic, impulsive, speech_dominant] (see its class_loss_weights
    # construction order).
    speech_dominant_index: int = 2

    # Below this confidence, the classifier isn't confident enough to
    # relax suppression at all -- full enhancement strength applies.
    # Above it, suppression strength relaxes linearly with confidence up
    # to 1.0. 0.6 is a reasoned starting point (majority-confident, not
    # just plurality-confident out of 3 classes), not a literature-
    # grounded constant -- validate against AEGIS's own eval curve, same
    # as every other tolerance introduced in this project's review passes.
    passthrough_confidence_threshold: float = 0.6

    # Enhancement strength never drops below this floor even at 100%
    # speech_dominant confidence. 0.3 means "at most 70% of suppression
    # strength can be relaxed" -- this is a suppression-strength floor,
    # not a bypass switch (see module docstring).
    min_enhancement_strength: float = 0.3

    # Exponential smoothing factor for the gate value across chunks
    # (compressor-style attack/release), in [0, 1). Higher = smoother /
    # slower-reacting (less pumping, more lag before relaxing or
    # re-tightening); lower = snappier / more prone to audible pumping.
    gate_smoothing: float = 0.85


class ConfidenceGatedEnhancer(nn.Module):
    """
    Wraps an SE model (DeepFilterNet3Wrapper / CleanUMambaWrapper) and the
    classifier (AudioClassifierNet), blending enhanced output with the
    original input based on the classifier's speech_dominant confidence.

    forward(x) -> (blended_output_or_tuple, diagnostics_dict)

    `blended_output_or_tuple` mirrors whatever calling convention the
    wrapped SE model used (see DeepFilterNet3Wrapper.forward's own
    docstring for its two conventions) -- only the waveform tensor
    participates in the blend; any returned state tensors pass through
    unchanged, so this wrapper is a drop-in addition around an existing
    SE model regardless of which streaming mode it's called in.

    `diagnostics_dict` carries the raw and smoothed gate values plus the
    classifier's confidence, both for runtime telemetry and so this
    module's behavior is directly unit-testable without needing a fully
    trained classifier (see the module-level test invocation in this
    file's companion test, which mocks classifier logits to check the
    blend math in isolation from classifier training quality).
    """

    def __init__(
        self,
        se_model: nn.Module,
        classifier: nn.Module,
        config: Optional[GatedEnhancementConfig] = None,
    ):
        super().__init__()
        self.se_model = se_model
        self.classifier = classifier
        self.config = config or GatedEnhancementConfig()
        self._smoothed_gate: Optional[float] = None

    def reset_state(self) -> None:
        """Resets both the wrapped SE model's streaming state and this
        wrapper's own gate smoothing -- call this between audio sessions,
        same as calling reset_state() on the SE model alone would have
        required before this wrapper existed."""
        if hasattr(self.se_model, "reset_state"):
            self.se_model.reset_state()
        self._smoothed_gate = None

    def _compute_gate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (raw_gate, speech_confidence). raw_gate is per-batch-item
        enhancement strength in [min_enhancement_strength, 1.0] -- 1.0
        means full enhancement (no relaxation), lower means more of the
        original signal is preserved in the blend.
        """
        cfg = self.config
        with torch.no_grad():
            logits = self.classifier(x)
            probs = F.softmax(logits, dim=-1)
            speech_conf = probs[:, cfg.speech_dominant_index]

        # Below threshold: no relaxation (gate = 1.0). Above it: relax
        # linearly from 1.0 down to min_enhancement_strength as confidence
        # rises from the threshold to 1.0.
        excess = torch.clamp(speech_conf - cfg.passthrough_confidence_threshold, min=0.0)
        max_excess = max(1.0 - cfg.passthrough_confidence_threshold, 1e-6)
        relax_fraction = torch.clamp(excess / max_excess, 0.0, 1.0)
        raw_gate = 1.0 - relax_fraction * (1.0 - cfg.min_enhancement_strength)
        return raw_gate, speech_conf

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, tuple], dict]:
        cfg = self.config
        se_out = self.se_model(x)

        if isinstance(se_out, tuple):
            # Explicit-state calling convention (see DeepFilterNet3Wrapper.
            # forward's docstring, mode 2) -- only the waveform participates
            # in the blend; state tensors pass through unchanged.
            enhanced_wave, *state_rest = se_out
        else:
            enhanced_wave, state_rest = se_out, None

        raw_gate, speech_conf = self._compute_gate(x)

        # Chunk-level exponential smoothing (compressor-style attack/
        # release) across forward() calls -- see module docstring for why
        # this matters given the classifier's coarser (0.2s) update rate
        # relative to the SE models' ~10ms streaming chunks.
        raw_gate_scalar = raw_gate.mean().item()
        if self._smoothed_gate is None:
            smoothed = raw_gate_scalar
        else:
            a = cfg.gate_smoothing
            smoothed = a * self._smoothed_gate + (1 - a) * raw_gate_scalar
        self._smoothed_gate = smoothed

        gate_tensor = torch.tensor(smoothed, dtype=enhanced_wave.dtype, device=enhanced_wave.device)
        blended = gate_tensor * enhanced_wave + (1.0 - gate_tensor) * x.reshape(enhanced_wave.shape)

        diagnostics = {
            "speech_dominant_confidence": speech_conf.mean().item(),
            "raw_gate": raw_gate_scalar,
            "smoothed_gate": smoothed,
        }

        if state_rest is not None:
            return (blended, *state_rest), diagnostics
        return blended, diagnostics

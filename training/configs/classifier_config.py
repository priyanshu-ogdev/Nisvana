"""
training/configs/classifier_config.py — Model 4: aegis-clf-gate

Lightweight SNR/harmonic-state classifier. Gating signal that routes
between Model 1 (primary, low-latency) and Model 2 (escalation) at
inference time, and feeds the harmonic-preprocessing / throat-mic fusion
logic described earlier in this design.

TUNING RATIONALE: no published reference architecture exists for this
exact task -- it's bespoke to this project, not a reproduction of a paper.
Every number below is a REASONED DEFAULT calibrated to the task's actual
scale, explicitly not a literature citation, and should move based on the
validation curve rather than being treated as fixed. This is stated
plainly in code, not just in a design doc, so it doesn't get mistaken for
a grounded number the way Models 1-3's config values are.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .base_config import BaseModelConfig
from training.callbacks.gradual_unfreezing import GradualUnfreezeConfig
from training.callbacks.ema import EmaConfig


# The 3-way collapse of the 10-class unified taxonomy, per the label
# crosswalk table from the data-forge review. This is the actual training
# target -- NOT the fine-grained unified_class list, which is used only
# for the SE branch's sampling weights, not as a classification label here.
GATE_CLASSES: List[str] = ["harmonic", "impulsive", "speech_dominant"]

UNIFIED_TO_GATE_CLASS: Dict[str, str] = {
    # Granular UnifiedClass keys from data_forge
    "tank_tracked": "harmonic",
    "artillery_howitzer": "harmonic",
    "jet_cockpit": "harmonic",
    "naval_destroyer": "harmonic",
    "military_vehicle": "harmonic",
    "drone_uav": "harmonic",
    "siren_emergency": "harmonic",
    "wind_rotor_gap": "harmonic",
    "general_noise": "harmonic",
    "explosion_blast": "impulsive",
    "gunshot_firearm": "impulsive",
    "clean_speech": "speech_dominant",
    # Broad training taxonomy aliases
    "gunfire": "impulsive",
    "explosion_artillery": "impulsive",
    "rotor_vehicle_drone": "harmonic",
    "armored_vehicle_naval_jet": "harmonic",
    "siren": "harmonic",           # sustained tonal sweep -- harmonic, not impulsive
    "wind": "harmonic",            # broadband but continuous/modulated, not transient
    "vehicle_engine_general": "harmonic",
    "babble_crowd": "speech_dominant",
    "broad_industrial": "harmonic",
    "speech_clean": "speech_dominant",
}

# Class weights for the loss function, inverse-frequency per the raw
# duration estimates established in the data-forge review: impulsive
# events (gunfire, explosions) are far rarer in raw seconds than harmonic/
# continuous noise, even though gunfire has many discrete real recordings
# -- a single gunshot event is ~0.1-1s, a single engine recording can run
# minutes. Without weighting, the classifier would just learn to predict
# "harmonic" most of the time and still score deceptively well on accuracy.
GATE_CLASS_LOSS_WEIGHTS: Dict[str, float] = {
    "impulsive": 3.0,
    "harmonic": 1.0,
    "speech_dominant": 1.2,
}


@dataclass
class ClassifierConfig(BaseModelConfig):
    model_key: str = "aegis-clf-gate"

    architecture: str = "small_gru"        # alt: "1d_cnn" -- either is reasonable at this task scale, GRU chosen
                                            # for slightly better handling of the harmonic/cyclostationary distinction
    input_features: str = "log_mel_64"     # 64-band log-mel, NOT raw waveform -- this is a coarse classification
                                            # task, doesn't need SE-scale feature resolution
    hidden_size: int = 128
    num_layers: int = 2

    pretrained_init: str = None            # trained from scratch -- no applicable pretrained checkpoint exists
                                            # for this bespoke task

    optimizer: str = "adamw"
    lr: float = 1e-3
    lr_schedule: str = "cosine"
    weight_decay: float = 1e-4

    batch_size: int = 192                  # larger than the SE models' -- classifier inputs (log-mel frames) are
                                            # far cheaper than full waveform triplets, no reason to keep batches small

    max_epochs: int = 25
    early_stopping_patience: int = 6       # small model, small task -- should converge fast; long training
                                            # risks overfitting the label-generation procedure itself, not the audio

    loss_fn: str = "cross_entropy"
    class_loss_weights: Dict[str, float] = field(default_factory=lambda: dict(GATE_CLASS_LOSS_WEIGHTS))

    # No separate dataset needed -- labels are generated for free from the
    # SAME mixing parameters used to build Models 1-3's SE shards (the
    # data-forge design decision this config depends on and should not
    # duplicate).
    label_source: str = "generated_from_mixing_params"

    # EMA disabled here, matching this module's own docstring claim: a
    # small, fast-converging model (25 epochs, patience 6) doesn't get
    # much benefit from smoothing a long noisy trajectory it doesn't have.
    ema: EmaConfig = field(default_factory=lambda: EmaConfig(enabled=False))

    # Also disabled: this model trains from scratch (pretrained_init=None,
    # already stated above) -- there is no pretrained checkpoint to protect
    # from catastrophic forgetting, so a freeze schedule has nothing to do.
    gradual_unfreeze: GradualUnfreezeConfig = field(default_factory=lambda: GradualUnfreezeConfig(enabled=False))

    # Integration test, not just a standalone accuracy number: does this
    # classifier's "impulsive/hard" flag actually correlate with where
    # Model 1 measurably underperforms Model 2? Tracked as a named metric,
    # not left implicit.
    validate_against_se_gap: bool = True

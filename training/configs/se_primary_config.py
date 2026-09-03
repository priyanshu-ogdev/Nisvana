"""
training/configs/se_primary_config.py — Model 1: aegis-se-primary

DeepFilterNet3, zero-lookahead variant. Always-on live enhancement path.

TUNING RATIONALE (not the paper's from-scratch numbers, deliberately):
The official DeepFilterNet3 config.ini trains FROM SCRATCH: lr=0.001,
120 epochs, batch 16->128 progressive. We warm-start from the public
fal/DeepFilterNet3 checkpoint instead -- the base speech/noise separation
capability is already learned; this run only needs to (a) adapt to
df_lookahead=0/conv_lookahead=0 (an architecture change, not just new data,
since it shrinks the DF module's receptive field) and (b) adapt to the
defence-specific noise classes layered into the data-forge shards. Both
justify a real fine-tune, not a token one -- hence LR at half the base
rate rather than the 10-20x reduction a pure data fine-tune would use,
and a shorter but non-trivial epoch budget.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .base_config import BaseModelConfig, UNIFIED_CLASSES


# Per-class oversampling factors for the WeightedRandomSampler, derived
# directly from the real-hours-per-class estimates established during the
# data-forge review. NOT a substitute for more real data on the thin
# classes -- a documented mitigation, with its own ceiling stated below.
CLASS_OVERSAMPLE_FACTORS: Dict[str, float] = {
    # Granular UnifiedClass keys from data_forge
    "tank_tracked": 6.0,
    "artillery_howitzer": 6.0,
    "jet_cockpit": 6.0,
    "naval_destroyer": 6.0,
    "military_vehicle": 6.0,
    "explosion_blast": 4.0,
    "gunshot_firearm": 1.0,
    "drone_uav": 1.0,
    "siren_emergency": 5.0,
    "wind_rotor_gap": 5.0,
    "clean_speech": 1.0,
    "general_noise": 1.0,
    # Broad training taxonomy aliases
    "gunfire": 1.0,                        # ~15-20h combined across NIJ/Kabealo/Cooper&Shaw/MAD/FSD50K -- abundant
    "rotor_vehicle_drone": 1.0,            # DroneAudioSet alone is 23.5h -- abundant, BUT see note below
    "vehicle_engine_general": 1.0,         # broad FSD50K/AudioSet/NOISEX-92 coverage -- abundant
    "babble_crowd": 1.0,
    "broad_industrial": 1.0,
    "speech_clean": 1.0,                   # DNS Challenge fullband -- effectively unlimited
    "explosion_artillery": 4.0,            # SHAReD (326 clips) + MAD partial + thin FSD50K -- scarce
    "armored_vehicle_naval_jet": 6.0,      # NOISEX-92, ~1h total across 7 platforms -- most scarce Tier-A source
    "siren": 5.0,                          # UrbanSound8K subset, under 1h
    "wind": 5.0,                           # MUSAN/FSD50K subset only, no dedicated corpus exists (confirmed absent)
}
# CEILING NOTE: factors above 6x are deliberately not used. Beyond that,
# oversampling stops helping generalization and starts risking memorization
# of the specific few real clips available -- this is a data-scarcity
# problem, not a hyperparameter problem, for armored_vehicle_naval_jet
# especially. The oversample factor buys "not starved during training,"
# not "as well-generalized as gunfire." Report this class's eval metrics
# with that caveat, per the data-forge review's disclosure requirement.

# CAUTION on rotor_vehicle_drone=1.0: DroneAudioSet's 23.5h covers DRONES
# specifically, not helicopters -- the two share this unified_class bucket
# but are acoustically distinct (rotor size/blade count/engine type). If
# per-class shard metadata distinguishes drone vs. helicopter sub-labels,
# override this to weight helicopter sub-clips at ~5.0 like the other thin
# classes; the flat 1.0 here assumes the bucket is drone-dominated by volume,
# which will under-expose the helicopter sub-case if left unexamined.


@dataclass
class DfLossConfig:
    """
    Multi-resolution spectral + local-SNR loss, values taken directly from
    the official DeepFilterNet3 config.ini -- NOT re-tuned, since loss
    weighting is tightly coupled to the pretrained checkpoint's learned
    scale and changing it during fine-tuning risks destabilizing weights
    that don't need to move.
    """
    multires_spec_factor: float = 500.0
    multires_spec_factor_complex: float = 500.0
    multires_spec_gamma: float = 0.3
    multires_fft_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    local_snr_factor: float = 1e-3


@dataclass
class SePrimaryConfig(BaseModelConfig):
    model_key: str = "aegis-se-primary"

    # Architecture deltas from the pretrained checkpoint -- this is the
    # one deliberate change from stock DeepFilterNet3.
    df_lookahead: int = 0          # stock default: 2 -- zeroed for the live path's latency budget
    conv_lookahead: int = 0        # stock default: 2

    nb_erb: int = 32               # unchanged from stock config.ini
    nb_df: int = 96                # unchanged from stock config.ini
    df_order: int = 5              # unchanged from stock config.ini
    conv_ch: int = 64              # unchanged from stock config.ini

    pretrained_init: str = "fal/DeepFilterNet3"   # warm-start, not from-scratch

    # Optimizer -- half the base LR (0.001 -> 0.0005) to reflect warm-start,
    # not a from-scratch run, while still being high enough to genuinely
    # adapt the lookahead-dependent layers, not just nudge them.
    optimizer: str = "adamw"
    lr: float = 5e-4
    lr_min: float = 1e-6
    weight_decay_start: float = 1e-12
    weight_decay_end: float = 0.01
    warmup_epochs: int = 2                        # shorter than stock's 3 -- warm start needs less ramp

    # Batch size: skip the stock config's low starting batch (16) since
    # we're not stabilizing from-scratch training; start where stock ends
    # its early ramp and grow further given fine-tune's shorter schedule.
    batch_size_schedule: Dict[int, int] = field(default_factory=lambda: {
        0: 32, 3: 64, 10: 128,
    })

    max_epochs: int = 50                          # vs. stock's 120 -- fine-tune, not from-scratch
    early_stopping_patience: int = 12

    max_sample_len_s: float = 3.0                 # unchanged from stock -- no reason to deviate

    # Training SNR set: stock's -100/-5/0/5/10/20/40. We drop -100 (the
    # "effectively clean" bucket) proportion slightly and lean toward the
    # PS's literal target range (>15dB SNR improvement implies training
    # exposure spanning well below and around that threshold).
    dataloader_snrs: List[int] = field(default_factory=lambda: [-100, -5, 0, 5, 10, 20, 40])
    dataloader_snr_weights: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.20, 0.20, 0.15, 0.10, 0.05])

    loss: DfLossConfig = field(default_factory=DfLossConfig)
    class_oversample_factors: Dict[str, float] = field(default_factory=lambda: dict(CLASS_OVERSAMPLE_FACTORS))

    eval_reference_target_snr_db: float = 15.0     # PS's literal target, tracked explicitly in eval logs
    eval_reference_target_stoi: float = 0.85
    eval_reference_target_pesq: float = 2.5

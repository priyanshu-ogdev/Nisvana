"""
training/schedulers/snr_curriculum.py — Optional easy-to-hard (or hard-to-easy)
SNR curriculum over the course of training.

GROUNDED, WITH TWO HONEST CAVEATS (the second added on review -- the
original docstring overstated the first citation's task-relevance):

1. SNR-Decaying Curriculum Learning (SDCL, arXiv:2510.18533) validates an
   easy-to-hard schedule -- start at higher SNR (easier), exponentially
   decay the sampling distribution's mean toward harder (lower) SNR as
   training progresses. That's the DEFAULT direction implemented below.
   CORRECTION FROM REVIEW: this paper's title is "Noise-Conditioned
   Mixture-of-Experts Framework for Robust Speaker Verification" (Gu et
   al.) -- SDCL is one component of a larger speaker-verification system,
   validated on VoxCeleb1 speaker-verification EER, NOT on speech
   enhancement PESQ/STOI/SNR. The original docstring here implied more
   direct task-relevance ("task-closer... speech, not just ASR-adjacent")
   than the paper actually provides -- speaker verification and speech
   enhancement are both "speech" tasks but optimize toward very different
   objectives (identity discrimination vs. waveform reconstruction), and
   a curriculum's interaction with the loss landscape is not guaranteed
   to transfer between them. The underlying SDCL mechanism is still worth
   testing here -- but as an idea borrowed across tasks, not a directly
   validated one, which if anything strengthens (not weakens) the
   "ablation, not a confident default" posture this module already takes.

2. An earlier study on ASR under noise (Braun et al., 2017, as surveyed
   in arXiv:2101.10382) found the OPPOSITE ordering won for their task --
   starting with the hardest (lowest SNR) examples first and easing up
   outperformed easy-to-hard, and they explicitly tested both directions
   before reporting this. This is genuinely mixed evidence across tasks,
   and neither citation here is a direct, same-task validation -- one is
   speaker verification (SDCL), the other is ASR (Braun et al.), and this
   module's actual task is speech enhancement. Direction is left as a
   config flag (`direction="easy_to_hard"` vs `"hard_to_easy"`), with
   `"easy_to_hard"` kept as the default only because it's the more
   recently published of two imperfect analogues, not because either has
   been shown to work for this specific task. BOTH directions should be
   tried against AEGIS's own validation curve -- treat the default as a
   starting point for an ablation, never as a settled recommendation.

DISABLED by default for Models 1-3's base training run below (see
`enabled: bool = False` in each SE config) -- curriculum scheduling is a
genuine additional axis of complexity on top of everything already tuned
in this design, and the honest position is that it should be validated as
an ablation, not shipped as an on-by-default assumption when the evidence
itself disagrees across tasks.
"""

from dataclasses import dataclass
from typing import List, Literal

import numpy as np


@dataclass
class SnrCurriculumConfig:
    enabled: bool = False              # opt-in, see docstring -- evidence is genuinely mixed
    direction: Literal["easy_to_hard", "hard_to_easy"] = "easy_to_hard"  # SDCL's validated direction, as a
                                                                          # starting point, not a settled answer
    start_mean_snr_db: float = 20.0    # matches data_forge's ForgeMixingConfig.max_snr_db (20.0) -- the easy end
    # REVIEW-PASS FIX: was 0.0. data_forge/config.py's real mixing range is
    # min_snr_db=-5.0 to max_snr_db=20.0 -- a curriculum floor of 0.0 meant
    # the hardest 5dB of SNR this project's own data pipeline actually
    # produces (-5 to 0dB, precisely the extreme-low-SNR region this
    # project's own research -- the Fraunhofer comparative study -- flagged
    # as the highest-risk range for real-time causal models) was NEVER
    # emphasized by the curriculum's mean-shifting schedule, at any epoch,
    # even at the end of training. Corrected to reach the data's actual
    # floor.
    end_mean_snr_db: float = -5.0      # matches data_forge's ForgeMixingConfig.min_snr_db -- the hard end
    decay_span_epochs: int = 30
    std_db: float = 8.0                # truncated-Gaussian spread around the epoch's mean, per SDCL's approach


def current_mean_snr(config: SnrCurriculumConfig, epoch: int) -> float:
    """Exponential decay of the sampling distribution's mean SNR toward the
    target, over `decay_span_epochs`, then holds steady."""
    if not config.enabled:
        raise RuntimeError("current_mean_snr called with curriculum disabled -- check config.enabled first.")

    progress = min(epoch / max(config.decay_span_epochs, 1), 1.0)
    if config.direction == "easy_to_hard":
        start, end = config.start_mean_snr_db, config.end_mean_snr_db
    else:
        start, end = config.end_mean_snr_db, config.start_mean_snr_db

    # Exponential (not linear) decay, matching SDCL's described schedule shape.
    decay_rate = 3.0  # steepness; higher = faster early transition
    frac = 1.0 - np.exp(-decay_rate * progress)
    frac = frac / (1.0 - np.exp(-decay_rate))  # normalize to reach `end` exactly at progress=1.0
    return start + (end - start) * frac


def sample_snr_for_epoch(config: SnrCurriculumConfig, epoch: int, rng: np.random.Generator) -> float:
    mean = current_mean_snr(config, epoch)
    # Truncated at +/- 2 std to avoid sampling absurd SNRs far outside the
    # dataloader_snrs range each model config already defines.
    return float(np.clip(rng.normal(mean, config.std_db), mean - 2 * config.std_db, mean + 2 * config.std_db))


# ==============================================================================
# Noise-Type Curriculum (Phase 4 SOTA Addition)
#
# Complements the SNR curriculum above: while the SNR curriculum controls
# HOW MUCH noise the model sees, this controls WHAT KIND of noise it sees
# at each stage of training.
#
# Rationale for real-recordings-only training: with limited real data per
# noise class (especially rare events like gunshots and explosions),
# introducing all classes simultaneously forces the model to spread its
# early learning capacity across classes with vastly different temporal
# statistics. Starting with easier stationary/harmonic noise builds stable
# spectral representations first, then progressively introduces harder
# transient events once the model has a solid foundation. This is
# analogous to how human listeners develop noise robustness — familiarity
# with steady-state noise precedes ability to process sudden impulsive events.
#
# Implementation: provides class-level sampling weights per epoch that the
# DataLoader's weighted sampler (weighted_shard_sampler.py) can consume.
# Does NOT filter out any classes — all classes remain in the training pool
# at all times (preventing complete blindness to any class), but their
# relative sampling probability is modulated by the curriculum phase.
# ==============================================================================

@dataclass
class NoiseTypeCurriculumConfig:
    """Controls progressive introduction of noise difficulty categories."""
    enabled: bool = False  # opt-in, same discipline as SNR curriculum

    # Phase boundaries (in epochs)
    phase_1_end_epoch: int = 10   # Stationary-dominant phase
    phase_2_end_epoch: int = 25   # Mixed phase (all classes weighted equally)
    # After phase_2_end_epoch: impulsive-boosted phase (emphasize hardest classes)

    # Noise classes grouped by difficulty for curriculum progression.
    # These must match the unified_class values in shard JSON metadata.
    stationary_classes: List[str] = None
    transient_classes: List[str] = None
    impulsive_classes: List[str] = None

    # Weight multipliers per phase for each difficulty tier
    # Phase 1: stationary=2.0, transient=0.5, impulsive=0.3
    # Phase 2: all=1.0 (uniform)
    # Phase 3: stationary=0.7, transient=1.0, impulsive=2.0
    phase_1_stationary_weight: float = 2.0
    phase_1_transient_weight: float = 0.5
    phase_1_impulsive_weight: float = 0.3
    phase_3_stationary_weight: float = 0.7
    phase_3_transient_weight: float = 1.0
    phase_3_impulsive_weight: float = 2.0

    def __post_init__(self):
        if self.stationary_classes is None:
            self.stationary_classes = [
                "tank_tracked", "artillery_howitzer", "jet_cockpit",
                "naval_destroyer", "military_vehicle", "drone_uav",
                "general_noise", "wind_rotor_gap",
                # Broad aliases
                "armored_vehicle_naval_jet", "rotor_vehicle_drone",
                "vehicle_engine_general", "broad_industrial",
            ]
        if self.transient_classes is None:
            self.transient_classes = [
                "siren_emergency", "siren",
                "babble_crowd",
            ]
        if self.impulsive_classes is None:
            self.impulsive_classes = [
                "explosion_blast", "gunshot_firearm",
                "explosion_artillery", "gunfire",
            ]


def get_noise_type_weight(
    config: NoiseTypeCurriculumConfig,
    unified_class: str,
    epoch: int,
) -> float:
    """
    Returns the curriculum sampling weight multiplier for a given noise class
    at a given epoch. This weight is MULTIPLIED with the sync_tier weight and
    class_oversample_factor already computed in weighted_shard_sampler.py —
    it's an additional axis, not a replacement.

    Returns 1.0 for all classes when curriculum is disabled or for classes
    not found in any difficulty tier.
    """
    if not config.enabled:
        return 1.0

    # Determine which difficulty tier this class belongs to
    if unified_class in config.stationary_classes:
        tier = "stationary"
    elif unified_class in config.transient_classes:
        tier = "transient"
    elif unified_class in config.impulsive_classes:
        tier = "impulsive"
    else:
        return 1.0  # Unknown class — no curriculum modulation

    # Phase 1: Stationary-dominant (easy start)
    if epoch < config.phase_1_end_epoch:
        weights = {
            "stationary": config.phase_1_stationary_weight,
            "transient": config.phase_1_transient_weight,
            "impulsive": config.phase_1_impulsive_weight,
        }
    # Phase 2: Uniform (all classes equally weighted)
    elif epoch < config.phase_2_end_epoch:
        weights = {"stationary": 1.0, "transient": 1.0, "impulsive": 1.0}
    # Phase 3: Impulsive-boosted (hard focus)
    else:
        weights = {
            "stationary": config.phase_3_stationary_weight,
            "transient": config.phase_3_transient_weight,
            "impulsive": config.phase_3_impulsive_weight,
        }

    return weights.get(tier, 1.0)


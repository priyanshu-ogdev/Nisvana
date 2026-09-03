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
    start_mean_snr_db: float = 20.0
    end_mean_snr_db: float = 0.0
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

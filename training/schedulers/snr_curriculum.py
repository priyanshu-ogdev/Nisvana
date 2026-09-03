"""
training/schedulers/snr_curriculum.py — Optional easy-to-hard (or hard-to-easy)
SNR curriculum over the course of training.

GROUNDED, WITH AN HONEST CAVEAT: SNR-Decaying Curriculum Learning (SDCL,
arXiv:2510.18533) validates an easy-to-hard schedule -- start at higher
SNR (easier), exponentially decay the sampling distribution's mean toward
harder (lower) SNR as training progresses. That's the DEFAULT direction
implemented below.

BUT: an earlier study on ASR under noise (Braun et al., 2017, as surveyed
in arXiv:2101.10382) found the OPPOSITE ordering won for their task --
starting with the hardest (lowest SNR) examples first and easing up
outperformed easy-to-hard, and they explicitly tested both directions
before reporting this. This is genuinely mixed evidence across tasks, not
a settled question -- direction is left as a config flag
(`direction="easy_to_hard"` vs `"hard_to_easy"`), defaulting to the more
recent, task-closer (speech, not just ASR-adjacent) SDCL result, but
BOTH should be tried against AEGIS's own validation curve before trusting
either blindly. Do not treat the default as a confident recommendation --
treat it as the more literature-supported starting point for an ablation.

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

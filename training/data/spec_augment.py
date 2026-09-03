"""
training/data/spec_augment.py — On-the-fly spectral masking at DataLoader time

DESIGN CHANGE from the data-forge augmentation policy: data-forge's
augmentation (level/time-stretch jitter on the thin defence classes) is
baked into stored audio, by design, since it's correcting for genuine data
scarcity that needs to exist before mixing. SpecMix/SpecAugment-style
masking is different in kind -- it's meant to vary EVERY epoch, not be a
fixed pre-computed transform, so it belongs at DataLoader time, applied
fresh to each batch, not stored to disk.

GROUNDED: the same augmentation-comparison study cited in the data-forge
review (arXiv:2602.14671, "Data Augmentation for Pathological Speech
Enhancement") found SpecMix gives moderate, real gains -- specifically
positioned as one tier below noise-augmentation's larger effect but above
generative augmentation's near-zero-to-harmful effect. That paper's
ranking is the reason this is added as a light touch (bounded mask sizes,
low probability), not a heavy one -- "moderate," not "largest," gains is
the claim being acted on here.

Applied ONLY to the noisy/mixture input, NEVER to the clean target -- the
same non-negotiable rule as the data-forge review's "never touch the
ground truth" principle for pitch-shifting, applied here to spectral
masking for the same underlying reason: corrupting the target corrupts
every downstream metric, not just this one training run.
"""

from dataclasses import dataclass
import random

import numpy as np


@dataclass
class SpecMixConfig:
    enabled: bool = True
    time_mask_max_frames: int = 8      # bounded -- large masks on 3s clips at 48kHz framing
                                        # risk masking out the entire transient in gunfire/
                                        # explosion samples, which would corrupt the very
                                        # signal the thin classes need the model to learn
    freq_mask_max_bins: int = 8
    num_time_masks: int = 2
    num_freq_masks: int = 2
    apply_probability: float = 0.5     # NOT applied to every batch -- moderate effect size
                                        # cited above doesn't justify making it the default
                                        # state of every training example


def apply_spec_mix(spectrogram: np.ndarray, config: SpecMixConfig) -> np.ndarray:
    """
    spectrogram: (freq_bins, time_frames) magnitude or log-mel array for the
    NOISY input only. Returns a masked copy; never call this on the clean
    target.
    """
    if not config.enabled or random.random() > config.apply_probability:
        return spectrogram

    out = spectrogram.copy()
    n_freq, n_time = out.shape

    for _ in range(config.num_freq_masks):
        width = random.randint(0, config.freq_mask_max_bins)
        if width == 0 or width >= n_freq:
            continue
        start = random.randint(0, n_freq - width)
        out[start:start + width, :] = 0.0

    for _ in range(config.num_time_masks):
        width = random.randint(0, config.time_mask_max_frames)
        if width == 0 or width >= n_time:
            continue
        start = random.randint(0, n_time - width)
        out[:, start:start + width] = 0.0

    return out

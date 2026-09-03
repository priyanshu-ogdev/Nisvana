"""
training/configs/se_escalation_config.py — Model 2: aegis-se-escalation

DeepFilterNet3, STOCK lookahead (2/2 frames) -- this is the escalation path
Model 4's classifier routes hard segments to, so the extra ~40ms latency is
acceptable here in exchange for better quality on genuinely hard segments.

TUNING RATIONALE: unlike Model 1, this is a PURE data-distribution fine-tune
-- no architecture change from the pretrained checkpoint (lookahead stays at
the stock value). That means a lighter touch is correct: lower LR, fewer
epochs, less risk of disturbing weights that don't need to move. The one
substantive change is the low-SNR-weighted sampling distribution, grounded
in Shetu, Habets & Brendel (arXiv:2408.14582): low-SNR-weighted training
improves performance across ALL SNR conditions, not just the hard ones --
this is why the SNR reweighting isn't reserved only for a "hard cases"
subset, it shifts the whole training distribution.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .base_config import BaseModelConfig
from .se_primary_config import DfLossConfig, CLASS_OVERSAMPLE_FACTORS


@dataclass
class SeEscalationConfig(BaseModelConfig):
    model_key: str = "aegis-se-escalation"

    # Architecture: UNCHANGED from stock -- this is the whole point of this
    # variant existing alongside Model 1.
    df_lookahead: int = 2
    conv_lookahead: int = 2
    nb_erb: int = 32
    nb_df: int = 96
    df_order: int = 5
    conv_ch: int = 64

    pretrained_init: str = "fal/DeepFilterNet3"

    # Optimizer: lighter than Model 1's -- pure data fine-tune, no
    # architecture delta to adapt to.
    optimizer: str = "adamw"
    lr: float = 2e-4
    lr_min: float = 1e-6
    weight_decay_start: float = 1e-12
    weight_decay_end: float = 0.01
    warmup_epochs: int = 1

    batch_size_schedule: Dict[int, int] = field(default_factory=lambda: {
        0: 64, 5: 128,
    })

    max_epochs: int = 25                           # shorter than Model 1 -- lighter fine-tune
    early_stopping_patience: int = 8

    max_sample_len_s: float = 3.0

    # THE key change vs. Model 1 / vs. stock: reweighted toward low SNR.
    # Stock's roughly-uniform 7-way SNR distribution shifted so -5/0dB
    # dominate, per the cited evidence. This should be coordinated with
    # data_forge's mixing-stage SNR range (currently -5 to 20dB) -- this
    # config reweights WITHIN that range rather than requiring a re-mix,
    # so Models 1-3 can all read the same shards.
    dataloader_snrs: List[int] = field(default_factory=lambda: [-100, -5, 0, 5, 10, 20, 40])
    dataloader_snr_weights: List[float] = field(default_factory=lambda: [0.03, 0.30, 0.25, 0.20, 0.15, 0.05, 0.02])

    loss: DfLossConfig = field(default_factory=DfLossConfig)
    class_oversample_factors: Dict[str, float] = field(default_factory=lambda: dict(CLASS_OVERSAMPLE_FACTORS))

    eval_reference_target_snr_db: float = 15.0
    eval_reference_target_stoi: float = 0.85
    eval_reference_target_pesq: float = 2.5

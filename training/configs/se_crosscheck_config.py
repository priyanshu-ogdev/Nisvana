"""
training/configs/se_crosscheck_config.py — Model 3: aegis-se-crosscheck

CleanUMamba. Independent architecture cross-check against Models 1-2, and
the GPU-fallback path if DeepFilterNet3's ONNX/TensorRT export runs long
on the target hardware.

TUNING RATIONALE: the CleanUMamba paper (arXiv:2410.11062) trains from
scratch for 1M iterations at batch 16, but ALSO documents a specific
post-pruning fine-tune recipe: 100K additional iterations at batch 16,
same optimizer settings. That documented fine-tune recipe -- not the
from-scratch one -- is what we base this config on, since we're starting
from a pretrained checkpoint the same way Models 1-2 do.

CRITICAL: the paper trains on "the DNS dataset" without specifying
fullband vs. 16kHz DNS-2020. Per the sync-tier review, this run MUST use
the same 48kHz shard set as Models 1-2, and the paper's own 16kHz
DNS-2020 PESQ/STOI numbers must NOT be cited as this model's baseline --
re-evaluate after fine-tuning, on our shards, at 48kHz, full stop.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .base_config import BaseModelConfig
from training.callbacks.gradual_unfreezing import GradualUnfreezeConfig
from .se_primary_config import CLASS_OVERSAMPLE_FACTORS


@dataclass
class SeCrosscheckConfig(BaseModelConfig):
    model_key: str = "aegis-se-crosscheck"

    # Architecture: time-domain U-Net, Mamba only in the bottleneck,
    # structured channel pruning applied (paper demonstrates 8x reduction
    # without quality loss). Parameter-count target chosen to match
    # Model 1's inference cost class for a fair fallback comparison.
    target_param_count: str = "1M"        # one of the paper's pruned variants: 200K/500K/1M/2M
    bottleneck_uses_mamba: bool = True

    pretrained_init: str = "cleanumamba-1M-dns-pretrained"  # placeholder key; swap for the actual released checkpoint id

    # Optimizer -- taken directly from the paper's documented POST-PRUNING
    # fine-tune recipe (Section V-D), not the from-scratch recipe.
    optimizer: str = "adam"
    lr: float = 2e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_warmup_fraction: float = 0.05       # linear warmup for first 5% of steps, then cosine decay -- paper Section IV-C

    batch_size: int = 16                   # unchanged from paper -- both the from-scratch AND fine-tune recipes use 16
    total_finetune_steps: int = 150_000    # paper's documented fine-tune length was 100K; +50% given our data mix
                                            # is more diverse (defence classes) than the paper's original DNS-only mix

    max_sample_len_s: float = 3.0          # matched to Models 1-2 for comparable eval

    loss_fn: str = "full_stft_loss"        # paper Section V-A

    mixed_precision: bool = True           # paper explicitly uses PyTorch AMP -- keep this, don't disable it

    # Same SNR/class weighting philosophy as Model 1 (the "primary" role
    # this model is meant to cross-check) -- NOT Model 2's low-SNR skew,
    # since the cross-check's job is validating Model 1's numbers on
    # matched data, not exploring a different training distribution.
    dataloader_snrs: List[int] = field(default_factory=lambda: [-100, -5, 0, 5, 10, 20, 40])
    dataloader_snr_weights: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.20, 0.20, 0.15, 0.10, 0.05])
    class_oversample_factors: Dict[str, float] = field(default_factory=lambda: dict(CLASS_OVERSAMPLE_FACTORS))

    eval_reference_target_snr_db: float = 15.0
    eval_reference_target_stoi: float = 0.85
    eval_reference_target_pesq: float = 2.5

    # Disabled here deliberately -- see gradual_unfreezing.py's docstring:
    # this model's pretrained_init may not resolve to a real checkpoint at
    # the target parameter count, in which case this is effectively a
    # from-scratch run and a freeze schedule would be actively wrong.
    # Confirm the actual initialization path before enabling.
    gradual_unfreeze: GradualUnfreezeConfig = field(default_factory=lambda: GradualUnfreezeConfig(enabled=False))

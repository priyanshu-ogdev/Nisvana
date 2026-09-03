"""
training/configs/aec_config.py — Model 5: aegis-aec-gate

Gated acoustic echo cancellation. Uses the existing deepvqe-ggml
reimplementation checkpoint AS-IS by default -- this config exists so the
option to fine-tune is well-defined if it's ever exercised, not because
training is expected to run in the base implementation plan.

Kept in training/configs/ for naming consistency (every model gets a
config, whether or not it trains by default) rather than special-cased
out of the package structure.
"""

from dataclasses import dataclass

from .base_config import BaseModelConfig


@dataclass
class AecGateConfig(BaseModelConfig):
    model_key: str = "aegis-aec-gate"

    train_by_default: bool = False         # explicit, load-bearing flag -- scripts/train_aec.py checks this
                                            # and refuses to run without --force, so nobody accidentally kicks
                                            # off a fine-tune run nobody asked for

    pretrained_checkpoint: str = "deepvqe-ggml-unofficial"   # used as-is when train_by_default is False

    # If ever revisited: the original DeepVQE paper (arXiv:2306.03177)
    # doesn't publish a full hyperparameter table the way DeepFilterNet3/
    # CleanUMamba do -- these are PLACEHOLDERS, not grounded values, and
    # are marked as such rather than presented with false confidence.
    # A literature pass specific to this fine-tune would be needed before
    # these numbers should be trusted.
    optimizer_placeholder: str = "adamw"
    lr_placeholder: float = 1e-4
    batch_size_placeholder: int = 16
    max_epochs_placeholder: int = 30

    data_source: str = "microsoft/aec-challenge"   # real+synthetic quadruplets, 48kHz native -- the one dataset
                                                    # in this whole catalog purpose-built for exactly this task

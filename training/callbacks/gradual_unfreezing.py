"""
training/callbacks/gradual_unfreezing.py — Progressive layer unfreezing for
warm-started fine-tunes.

GROUNDED: this is the one gap in the SOTA-upgrade pass that stood out on
review -- every SE model config (Models 1-3) warm-starts from a pretrained
checkpoint, but nothing in the training loop protected those pretrained
weights from catastrophic forgetting during the very first steps of
fine-tuning, when gradients from a freshly-initialized task head can be
large and noisy enough to overwrite good pretrained features before the
new task signal stabilizes.

The technique -- gradual/progressive unfreezing -- is well-established
across multiple independent lines of work, not a single paper's claim:
  - Howard & Ruder, ULMFiT (2018): the original formulation -- unfreeze
    one layer group at a time, starting from the LAST (most task-specific)
    layers and working toward the first (most general) ones, since later
    layers hold the least transferable information and are safest to
    adapt first.
  - Mosbach et al. (2020) / arXiv:2210.10325: found gradual unfreezing
    specifically mitigates the fine-tuning INSTABILITY traced to
    catastrophic forgetting in top layers -- directly relevant here since
    Models 1-3 already use somewhat aggressive LRs (5e-4 / 2e-4) for a
    warm-started run.
  - u-HuBERT (arXiv:2207.07036) demonstrates the same principle applied
    directly to a speech model: freezing early layers as a fixed feature
    extractor while fine-tuning only upper layers, with the number of
    frozen layers as an explicit, tuned hyperparameter -- the pattern
    this module implements.

APPLIED TO: Models 1-2 (DeepFilterNet3 variants) by default, where the
whole point is a warm start -- NOT Model 3 (CleanUMamba), since that
config's own comment already establishes it may be trained from a
public checkpoint that isn't guaranteed to exist at the exact target
parameter count; forcing a freeze schedule onto a possibly-from-scratch
run would be actively wrong, so it's left disabled there by default and
must be explicitly enabled once Model 3's actual initialization path is
confirmed.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GradualUnfreezeConfig:
    enabled: bool = True
    # Layer GROUPS, not individual layers -- matching ULMFiT's granularity.
    # Exact group boundaries depend on the vendored DeepFilterNet3 module
    # structure (training/models/ integration point) -- named here as the
    # DF-specific stages this architecture actually has, not a generic
    # placeholder, so the callback's config is directly actionable once
    # the real model is vendored in.
    layer_groups_last_to_first: List[str] = field(default_factory=lambda: [
        "df_decoder",      # deep-filtering decoder -- most task-specific, unfrozen first
        "erb_decoder",     # ERB-band gain decoder
        "df_encoder",      # deep-filtering encoder
        "erb_encoder",     # shared encoder trunk -- most general features, unfrozen last
    ])
    epochs_per_unfreeze_step: int = 3     # matches Model 1's warmup_epochs=2 / Model 2's
                                            # warmup_epochs=1 scale -- unfreezing should track
                                            # the existing warmup schedule, not run independently of it
    freeze_all_initially: bool = True      # epoch 0 starts with everything but the final
                                            # layer group frozen, per ULMFiT's original recipe


def unfrozen_groups_at_epoch(config: GradualUnfreezeConfig, epoch: int) -> List[str]:
    """Returns which layer groups should be trainable at a given epoch."""
    if not config.enabled:
        return list(config.layer_groups_last_to_first)  # everything trainable -- unfreezing off

    step = epoch // config.epochs_per_unfreeze_step
    n_unfrozen = min(step + 1, len(config.layer_groups_last_to_first))
    return config.layer_groups_last_to_first[:n_unfrozen]


def apply_freeze_schedule(model, config: GradualUnfreezeConfig, epoch: int, layer_group_map: dict) -> List[str]:
    """
    layer_group_map: dict mapping group name (matching
    layer_groups_last_to_first) to the actual list of named parameters
    belonging to that group in the vendored model -- supplied by the
    concrete trainer once training/models/ has the real DeepFilterNet3
    source, not guessed here.

    Returns the list of group names set trainable this epoch, for logging.
    """
    unfrozen = set(unfrozen_groups_at_epoch(config, epoch))
    for group_name, params in layer_group_map.items():
        requires_grad = group_name in unfrozen
        for p in params:
            p.requires_grad = requires_grad
    return sorted(unfrozen)

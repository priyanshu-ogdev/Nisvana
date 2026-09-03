"""
training/callbacks/ema.py — Exponential Moving Average of model weights

GROUNDED, not a guess -- but with a review-pass addition that changes the
confidence level, not the mechanism: EMA is well-established for training
stability and generalization in deep learning generally (Kaizen, Kong et
al., arXiv:2106.07759 -- applied directly to speech recognition training
and found stability improves as decay approaches the high end of the
usual range, with divergence at low decay values and high update
frequency). A 2024 TMLR systematic study (arXiv:2411.18704, Morales-
Brotons, Vogels & Hendrikx -- verified on review, correctly represented)
gives the standard operating range: decay in [0.9, 0.9999], task-
dependent, with the note that EMA requires LESS learning-rate decay than
raw SGD/Adam iterates since averaging itself provides implicit
regularization -- worth knowing before assuming the existing per-model LR
schedules need re-tuning alongside this addition.

REVIEW-PASS ADDITION, task-specific and directly relevant: neither citation
above is a speech-enhancement study -- Kaizen is speech *recognition*,
the TMLR paper is deep learning broadly (image/language-model-heavy
experiments). A speech-enhancement-specific study found since (arXiv:
2505.05216, "Do We Need EMA for Diffusion-Based Speech Enhancement?")
reports the OPPOSITE of the general-purpose guidance above for their
setting: "unlike in image generation, short or absent EMA consistently
yields better speech enhancement performance." One real caveat on that
finding's own applicability here -- it's diffusion-based SE, and Models
1-3 in this design are discriminative, not diffusion, so the finding
doesn't transfer with full force. But it is the only same-*task*
(speech enhancement) EMA evidence found across this project's research,
and it disagrees with the general-purpose default below. This is enough
to downgrade EMA from "grounded default" to the same posture already
applied to the SNR curriculum: kept enabled below as the more literature-
supported starting point across imperfect analogues, but explicitly
flagged as needing an on/off ablation against AEGIS's own validation
curve before being trusted, not assumed to help because the general deep-
learning literature says weight-averaging usually does.

Applied to Models 1-3 (the SE branch) by default -- as an ablation-worthy
default per the above, not a settled one. NOT applied to Model 4
(the classifier) by default -- it's a small, fast-converging model per its
own config's reasoning (25 epochs, early-stopping patience 6), and EMA's
main benefit (smoothing long, noisy training trajectories) is weaker where
the trajectory is already short. Left as an opt-in flag, not forced.
"""

from copy import deepcopy
from dataclasses import dataclass


@dataclass
class EmaConfig:
    enabled: bool = True          # ablation-worthy default, not a settled one -- see docstring's
                                   # review-pass addition (arXiv:2505.05216 found the opposite
                                   # result for speech enhancement specifically). Run with this
                                   # both True and False on Model 1 before trusting either.
    decay: float = 0.999          # mid-range per the cited TMLR guidance ([0.9, 0.9999]) --
                                   # not the most aggressive (0.9999) since these are fine-tunes
                                   # of a few tens of epochs, not from-scratch runs over hundreds;
                                   # a very slow-moving EMA wouldn't fully converge in that budget
    update_every_n_steps: int = 1  # per Kaizen's finding: high decay + Δ=1 was the stable
                                    # regime tested; sparser updates (Δ=10+) were only needed
                                    # to rescue LOW decay values, which we aren't using here
    use_for_eval: bool = True      # eval/checkpoint-selection reads the EMA weights, not raw
    use_for_final_export: bool = True  # inference engines load the EMA weights, per the cited
                                        # finding that EMA solutions generalize better than
                                        # last-iterate solutions, not just look smoother in-loop


class EmaTracker:
    """Maintains a shadow copy of model parameters, updated per EmaConfig."""

    def __init__(self, model, config: EmaConfig):
        self.config = config
        self.shadow = deepcopy(model.state_dict()) if config.enabled else None
        self._step = 0

    def update(self, model):
        if not self.config.enabled:
            return
        self._step += 1
        if self._step % self.config.update_every_n_steps != 0:
            return
        d = self.config.decay
        model_state = model.state_dict()
        for k, v in self.shadow.items():
            self.shadow[k] = d * v + (1.0 - d) * model_state[k]

    def state_dict(self):
        return self.shadow

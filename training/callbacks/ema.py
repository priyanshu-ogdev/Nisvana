"""
training/callbacks/ema.py — Exponential Moving Average of model weights

GROUNDED, not a guess: EMA is well-established for training stability and
generalization (Kaizen, Kong et al., arXiv:2106.07759 -- applied directly
to speech recognition training and found stability improves as decay
approaches the high end of the usual range, with divergence at low decay
values and high update frequency). A 2024 TMLR systematic study
(arXiv:2411.18704) gives the standard operating range: decay in
[0.9, 0.9999], task-dependent, with the note that EMA requires LESS
learning-rate decay than raw SGD/Adam iterates since averaging itself
provides implicit regularization -- worth knowing before assuming the
existing per-model LR schedules need re-tuning alongside this addition.

Applied to Models 1-3 (the SE branch) by default. NOT applied to Model 4
(the classifier) by default -- it's a small, fast-converging model per its
own config's reasoning (25 epochs, early-stopping patience 6), and EMA's
main benefit (smoothing long, noisy training trajectories) is weaker where
the trajectory is already short. Left as an opt-in flag, not forced.
"""

from copy import deepcopy
from dataclasses import dataclass


@dataclass
class EmaConfig:
    enabled: bool = True
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

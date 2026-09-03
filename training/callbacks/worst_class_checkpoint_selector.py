"""
training/callbacks/worst_class_checkpoint_selector.py

PROJECT-SPECIFIC DESIGN DECISION -- NOT a literature citation, stated
plainly so it isn't mistaken for a grounded technique the way EMA/SpecMix/
SNR-curriculum above are. This exists because of a problem specific to
AEGIS's own data situation, established across the data-forge review:
rotor/helicopter and wind are disclosed-thin classes with no further data
available. An aggregate-metric-only checkpoint selector (the default in
nearly every training framework) can silently pick a checkpoint that
improved the aggregate PESQ/STOI/SNR by trading away performance on
exactly the two classes already flagged as fragile -- because they're a
small fraction of total validation samples, an aggregate metric can barely
move while those two classes get meaningfully worse.

This callback tracks per-`unified_class` validation metrics (the
per-class breakdown already required by the evaluation protocol) and
selects/early-stops based on a combined criterion: aggregate metric AND
no regression beyond a tolerance on the worst-performing disclosed-weak
class, not aggregate alone.

REVIEW-PASS FIX #1 (was a real bug, not a style choice): `best_per_class`
was previously only updated inside the `if aggregate_improved:` branch.
That meant if a watched class's own metric improved on a step where the
*aggregate* did not, the improvement was never "banked" -- a later step
could regress below that unbanked high-water mark without ever tripping
the regression guard, because the comparison was still against a stale,
lower `prior_best`. `best_per_class` is now tracked independently per
watched class, updated every time that class's metric is seen, regardless
of whether the aggregate also improved on that step.

REVIEW-PASS FIX #2 (a real coverage gap, not a bug): this callback only
ever watched PESQ regression. SNR is the metric this whole project's own
research repeatedly found to be the highest-risk, least-evidenced target
(no direct SI-SDR/SNR citation was ever found for the SE-branch backbone
across the design's own review history, unlike PESQ/STOI which both have
real cited margins). A selector that guards PESQ but not SNR on exactly
the two disclosed-weak classes is watching the wrong risk for this
project's actual failure mode. Both `pesq_<class>` and `snr_<class>` are
now watched per class, each with their own independently configurable
tolerance -- SNR's tolerance is expressed in dB, not the PESQ MOS-like
scale, so they are deliberately separate fields, not one shared number.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# The two classes carrying the standing disclosure from the data-forge
# review. If a future data pass genuinely closes one of these gaps, remove
# it here -- don't leave a class permanently flagged past the point it's
# actually still weak.
DISCLOSED_WEAK_CLASSES: List[str] = ["rotor_vehicle_drone", "wind"]

# Which per-class metrics this selector guards, and the eval-dict key
# prefix each one is read from (e.g. "pesq_wind", "snr_wind"). Extending
# this list (rather than hard-coding "pesq_" as the only prefix, as the
# pre-fix version did) is what makes adding SNR a one-line change instead
# of a second parallel code path.
WATCHED_METRIC_PREFIXES: Dict[str, str] = {
    "pesq": "pesq_",
    "snr": "snr_",
}


@dataclass
class WorstClassCheckpointConfig:
    enabled: bool = True
    watched_classes: List[str] = field(default_factory=lambda: list(DISCLOSED_WEAK_CLASSES))
    # A watched class's PESQ may not drop more than this from its own
    # true best-so-far, even if the aggregate metric improves.
    max_allowed_regression_pesq: float = 0.05
    # SNR's own regression tolerance, in dB -- deliberately not reusing
    # the PESQ number, since the two metrics are on unrelated scales.
    # 1.0dB is a conservative starting tolerance; treat it the same way
    # every other reasoned-default in this design is treated -- a
    # starting point to validate against AEGIS's own eval curve, not a
    # literature-grounded constant (no equivalent published guidance for
    # this specific selector-tolerance question was found).
    max_allowed_regression_snr_db: float = 1.0
    aggregate_metric_name: str = "pesq_aggregate"


class WorstClassCheckpointSelector:
    def __init__(self, config: WorstClassCheckpointConfig):
        self.config = config
        self.best_aggregate: Optional[float] = None
        # Separate high-water-mark dicts per watched metric (pesq/snr),
        # not one dict keyed by class alone -- the two metrics have
        # independent tolerances and must not be conflated.
        self.best_per_class: Dict[str, Dict[str, float]] = {"pesq": {}, "snr": {}}

    def _tolerance_for(self, metric: str) -> float:
        return (
            self.config.max_allowed_regression_pesq
            if metric == "pesq"
            else self.config.max_allowed_regression_snr_db
        )

    def should_accept_checkpoint(self, eval_metrics: Dict[str, float]) -> bool:
        """
        eval_metrics must include the aggregate metric plus, for every
        watched class, f"pesq_{unified_class}" and (where available)
        f"snr_{unified_class}" -- the per-class breakdown the evaluation
        protocol already requires, consumed here rather than duplicated.
        """
        if not self.config.enabled:
            return self._accept_aggregate_only(eval_metrics)

        aggregate = eval_metrics[self.config.aggregate_metric_name]
        aggregate_improved = self.best_aggregate is None or aggregate > self.best_aggregate

        # --- Regression check, both watched metrics, every step, ---
        # --- independent of whether the aggregate also improved.   ---
        # This is FIX #1: `best_per_class` is read (never written) in this
        # loop, so a class's own best-ever value can only ever go up,
        # regardless of what the aggregate did on any given step.
        for cls in self.config.watched_classes:
            for metric, prefix in WATCHED_METRIC_PREFIXES.items():
                key = f"{prefix}{cls}"
                if key not in eval_metrics:
                    continue  # class/metric not present in this eval batch; don't block on missing data
                current = eval_metrics[key]
                prior_best = self.best_per_class[metric].get(cls, current)
                regression = prior_best - current
                if regression > self._tolerance_for(metric):
                    # Aggregate may look better, but a disclosed-weak class
                    # regressed past tolerance on THIS metric -- reject
                    # this checkpoint as the new "best," even though a
                    # naive aggregate-only (or PESQ-only, pre-fix) selector
                    # would have accepted it.
                    return False

        # --- Bookkeeping: update every watched class/metric's true ---
        # --- best-so-far unconditionally, not gated on aggregate_improved. ---
        # This is the rest of FIX #1 -- an improvement on a step where the
        # aggregate didn't move is still a real improvement and must be
        # banked, or a later regression from that point is invisible to
        # the guard above.
        for cls in self.config.watched_classes:
            for metric, prefix in WATCHED_METRIC_PREFIXES.items():
                key = f"{prefix}{cls}"
                if key in eval_metrics:
                    d = self.best_per_class[metric]
                    d[cls] = max(d.get(cls, eval_metrics[key]), eval_metrics[key])

        if aggregate_improved:
            self.best_aggregate = aggregate
            return True
        return False

    def _accept_aggregate_only(self, eval_metrics: Dict[str, float]) -> bool:
        aggregate = eval_metrics[self.config.aggregate_metric_name]
        if self.best_aggregate is None or aggregate > self.best_aggregate:
            self.best_aggregate = aggregate
            return True
        return False

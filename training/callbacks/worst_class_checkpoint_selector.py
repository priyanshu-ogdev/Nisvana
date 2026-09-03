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
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# The two classes carrying the standing disclosure from the data-forge
# review. If a future data pass genuinely closes one of these gaps, remove
# it here -- don't leave a class permanently flagged past the point it's
# actually still weak.
DISCLOSED_WEAK_CLASSES: List[str] = ["rotor_vehicle_drone", "wind"]


@dataclass
class WorstClassCheckpointConfig:
    enabled: bool = True
    watched_classes: List[str] = field(default_factory=lambda: list(DISCLOSED_WEAK_CLASSES))
    max_allowed_regression_pesq: float = 0.05   # a watched class's PESQ may not drop more than
                                                  # this from its own best-so-far, even if the
                                                  # aggregate metric improves
    aggregate_metric_name: str = "pesq_aggregate"


class WorstClassCheckpointSelector:
    def __init__(self, config: WorstClassCheckpointConfig):
        self.config = config
        self.best_aggregate: Optional[float] = None
        self.best_per_class: Dict[str, float] = {}

    def should_accept_checkpoint(self, eval_metrics: Dict[str, float]) -> bool:
        """
        eval_metrics must include the aggregate metric plus
        f"pesq_{unified_class}" for every watched class -- this is the
        per-class breakdown the evaluation protocol already requires,
        consumed here rather than duplicated.
        """
        if not self.config.enabled:
            return self._accept_aggregate_only(eval_metrics)

        aggregate = eval_metrics[self.config.aggregate_metric_name]
        aggregate_improved = self.best_aggregate is None or aggregate > self.best_aggregate

        for cls in self.config.watched_classes:
            key = f"pesq_{cls}"
            if key not in eval_metrics:
                continue  # class not present in this eval batch; don't block on missing data
            current = eval_metrics[key]
            prior_best = self.best_per_class.get(cls, current)
            regression = prior_best - current
            if regression > self.config.max_allowed_regression_pesq:
                # Aggregate may look better, but a disclosed-weak class
                # regressed past tolerance -- reject this checkpoint as
                # the new "best," even though a naive aggregate-only
                # selector would have accepted it.
                return False

        if aggregate_improved:
            self.best_aggregate = aggregate
            for cls in self.config.watched_classes:
                key = f"pesq_{cls}"
                if key in eval_metrics:
                    self.best_per_class[cls] = max(self.best_per_class.get(cls, eval_metrics[key]), eval_metrics[key])
            return True
        return False

    def _accept_aggregate_only(self, eval_metrics: Dict[str, float]) -> bool:
        aggregate = eval_metrics[self.config.aggregate_metric_name]
        if self.best_aggregate is None or aggregate > self.best_aggregate:
            self.best_aggregate = aggregate
            return True
        return False

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from benches._shared.eval import roc_auc

__all__ = [
    "GOAL_RECALL_BAND",
    "GOAL_TARGET_AI_RECALL",
    "CandidateEvaluation",
    "GoalCandidate",
    "OperatingPoint",
    "SelectionResult",
    "operating_point_at_recall",
    "select_by_goal_criterion",
]

# Predefined, dev-only selection criterion for the placement sweeps: minimize human false flags
# at an acceptable AI recall, using the midpoint of the owner's ~80-90% tradeoff band.
GOAL_TARGET_AI_RECALL: Final = 0.85
GOAL_RECALL_BAND: Final = (0.80, 0.90)


@dataclass(frozen=True, slots=True)
class GoalCandidate:
    """One config's dev scores; AI-generated is the positive class (label 1)."""

    name: str
    labels: Sequence[int]
    scores: Sequence[float]


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    ai_recall: float
    human_fpr: float
    separation_margin: float


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    name: str
    target_recall: float
    threshold: float
    ai_recall: float
    human_fpr: float
    separation_margin: float
    roc_auc: float


@dataclass(frozen=True, slots=True)
class SelectionResult:
    target_recall: float
    ranked: tuple[CandidateEvaluation, ...]
    selected: str

    @property
    def selected_evaluation(self) -> CandidateEvaluation:
        return self.ranked[0]


def operating_point_at_recall(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    target_recall: float = GOAL_TARGET_AI_RECALL,
) -> OperatingPoint:
    """Return the dev operating point that first meets ``target_recall``.

    Recall and FPR are both non-increasing in the threshold, so the largest score threshold that
    still reaches the target recall is the one with the smallest human FPR. ``separation_margin``
    is that threshold minus the highest human score: a larger margin means human tracks sit
    further from the decision boundary, which is the goal when FPR is already tied at zero.
    """

    if not math.isfinite(target_recall) or not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall!r}")
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    y = list(labels)
    s = [float(value) for value in scores]
    if not any(label == 1 for label in y) or not any(label == 0 for label in y):
        raise ValueError("target-recall selection requires both classes in the evaluation data")
    if not all(math.isfinite(value) for value in s):
        raise ValueError("scores must be finite for target-recall selection")

    n_pos = sum(1 for label in y if label == 1)
    n_neg = len(y) - n_pos

    def reaches_target(threshold: float) -> bool:
        hits = sum(
            1 for value, label in zip(s, y, strict=True) if label == 1 and value >= threshold
        )
        return hits / n_pos + 1e-12 >= target_recall

    eligible = [threshold for threshold in sorted(set(s)) if reaches_target(threshold)]
    if not eligible:
        raise ValueError(f"no threshold reaches target recall {target_recall}")
    threshold = eligible[-1]
    tp = sum(1 for value, label in zip(s, y, strict=True) if label == 1 and value >= threshold)
    fp = sum(1 for value, label in zip(s, y, strict=True) if label == 0 and value >= threshold)
    human_max = max(value for value, label in zip(s, y, strict=True) if label == 0)
    return OperatingPoint(
        threshold=threshold,
        ai_recall=tp / n_pos,
        human_fpr=fp / n_neg,
        separation_margin=threshold - human_max,
    )


def select_by_goal_criterion(
    candidates: Sequence[GoalCandidate],
    *,
    target_recall: float = GOAL_TARGET_AI_RECALL,
) -> SelectionResult:
    """Rank dev-only candidates by the predefined goal-aligned criterion and pick the first.

    Lexicographic key, all computed on dev data only: human FPR at the target recall (lower is
    better), separation margin (higher is better), ROC-AUC (higher is better), then config name
    for a deterministic final tie-break. Held-out test scores are never passed to this function.
    """

    if not candidates:
        raise ValueError("select_by_goal_criterion requires at least one candidate")
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("duplicate candidate names are not allowed for deterministic selection")

    evaluations = []
    for candidate in candidates:
        point = operating_point_at_recall(
            candidate.labels, candidate.scores, target_recall=target_recall
        )
        evaluations.append(
            CandidateEvaluation(
                name=candidate.name,
                target_recall=target_recall,
                threshold=point.threshold,
                ai_recall=point.ai_recall,
                human_fpr=point.human_fpr,
                separation_margin=point.separation_margin,
                roc_auc=roc_auc(candidate.labels, candidate.scores),
            )
        )
    ranked = tuple(
        sorted(
            evaluations,
            key=lambda item: (
                item.human_fpr,
                -item.separation_margin,
                -item.roc_auc,
                item.name,
            ),
        )
    )
    return SelectionResult(target_recall=target_recall, ranked=ranked, selected=ranked[0].name)

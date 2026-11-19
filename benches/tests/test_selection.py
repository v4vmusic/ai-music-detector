from __future__ import annotations

import pytest

from benches._shared.eval import fpr_at_recall
from benches._shared.selection import (
    GOAL_TARGET_AI_RECALL,
    GoalCandidate,
    operating_point_at_recall,
    select_by_goal_criterion,
)


def _candidate(name: str, humans: list[float], ai: list[float]) -> GoalCandidate:
    return GoalCandidate(
        name=name,
        labels=[0] * len(humans) + [1] * len(ai),
        scores=[*humans, *ai],
    )


def test_operating_point_uses_highest_threshold_meeting_target_recall() -> None:
    humans = [0.10, 0.20, 0.30, 0.40]
    ai = [0.05, 0.06] + [0.5] * 14
    labels = [0] * len(humans) + [1] * len(ai)

    point = operating_point_at_recall(labels, [*humans, *ai], target_recall=0.85)

    assert point.threshold == pytest.approx(0.5)
    assert point.ai_recall == pytest.approx(14 / 16)
    assert point.human_fpr == pytest.approx(0.0)
    assert point.separation_margin == pytest.approx(0.1)
    # The goal operating point agrees with the existing FPR-at-recall helper.
    assert point.human_fpr == pytest.approx(fpr_at_recall(labels, [*humans, *ai], 0.85))


@pytest.mark.parametrize("target", [0.0, 1.01, float("nan")])
def test_operating_point_rejects_invalid_target_recall(target: float) -> None:
    with pytest.raises(ValueError, match="target_recall"):
        operating_point_at_recall([0, 1], [0.1, 0.9], target_recall=target)


def test_operating_point_rejects_missing_class_or_nonfinite_score() -> None:
    with pytest.raises(ValueError, match="both classes"):
        operating_point_at_recall([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        operating_point_at_recall([0, 1], [0.1, float("inf")])


def test_selection_prefers_lowest_fpr_then_larger_margin() -> None:
    candidates = [
        _candidate("AST-1c", [0.1, 0.2, 0.3, 0.4], [0.5] * 16),  # fpr 0, margin 0.1
        _candidate("AST-1b", [0.1, 0.2, 0.3, 0.4], [0.7] * 16),  # fpr 0, margin 0.3
        _candidate("AST-1a", [0.1, 0.2, 0.3, 0.8], [0.5] * 16),  # fpr 0.25
    ]

    result = select_by_goal_criterion(candidates, target_recall=0.85)

    assert result.target_recall == pytest.approx(0.85)
    assert result.selected == "AST-1b"
    assert tuple(item.name for item in result.ranked) == ("AST-1b", "AST-1c", "AST-1a")
    assert result.ranked[0].human_fpr == pytest.approx(0.0)
    assert result.ranked[0].separation_margin == pytest.approx(0.3)


def test_selection_breaks_fpr_and_margin_ties_by_roc_auc() -> None:
    humans = [0.1, 0.2, 0.3, 0.4]
    separable_ai = [0.41, 0.42, 0.43] + [0.5] * 17  # ROC-AUC 1.0
    lower_ai = [0.0, 0.01, 0.02] + [0.5] * 17  # three AI below humans

    result = select_by_goal_criterion(
        [
            _candidate("Z-1c", humans, lower_ai),
            _candidate("A-1b", humans, separable_ai),
        ]
    )

    # Both have fpr 0 and margin 0.1; the higher-AUC config wins.
    assert result.ranked[0].name == "A-1b"
    assert result.ranked[0].roc_auc > result.ranked[1].roc_auc
    assert result.selected == "A-1b"


def test_selection_final_name_tie_break_is_deterministic() -> None:
    humans = [0.1, 0.2]
    ai = [0.5, 0.6]

    result = select_by_goal_criterion(
        [_candidate("Z-1c", humans, ai), _candidate("A-1a", humans, ai)]
    )

    assert result.selected == "A-1a"


def test_default_target_recall_is_the_predefined_goal_value() -> None:
    assert GOAL_TARGET_AI_RECALL == pytest.approx(0.85)


def test_selection_rejects_empty_or_duplicate_candidate_names() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_by_goal_criterion([])
    with pytest.raises(ValueError, match="duplicate"):
        select_by_goal_criterion([_candidate("X", [0.1], [0.9]), _candidate("X", [0.2], [0.8])])

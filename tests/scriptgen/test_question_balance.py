"""Label balancing over the questions a rendered collection compiled."""

from __future__ import annotations

from typing import Any

from spatial_episode.scriptgen.question_balance import balance_question_labels


def _group(group_id: str, questions: list[tuple[str, str | None]]) -> dict[str, Any]:
    return {
        "question_group_id": group_id,
        "questions": [
            {
                "capability": capability,
                "role": "primary",
                "label": label,
                "skip_reason": None if label is not None else "invalid:clause:x",
            }
            for capability, label in questions
        ],
    }


def _stratum(report: dict[str, Any], capability: str) -> dict[str, Any]:
    return next(row for row in report["strata"] if row["capability"] == capability)


def test_every_retained_label_ships_the_same_number_of_questions() -> None:
    """The shipped stratum is flat, so no constant answer beats chance."""
    groups = [
        _group(f"g{index}", [("reference_frame_visibility_yaw90", label)])
        for index, label in enumerate(["visible"] * 2 + ["not_visible"] * 8)
    ]
    report = balance_question_labels(groups)
    stratum = _stratum(report, "reference_frame_visibility_yaw90")
    assert stratum["label_counts"] == {"visible": 2, "not_visible": 8}
    assert stratum["retained_per_label"] == 2
    assert stratum["retained"] == 4
    assert stratum["dropped"] == 6
    assert stratum["chance_accuracy"] == 0.5
    assert report["retained_count"] == 4


def test_a_stratum_with_one_answer_ships_nothing_and_says_so() -> None:
    """The k1 anchor question is pinned to "front" by the chain geometry.

    Downsampling cannot rescue it: a stratum with a single label teaches the
    prior, so it is dropped whole and named in the report rather than left to
    look like a stratum that simply produced few questions.
    """
    groups = [
        _group(f"g{index}", [("cross_view_snapshot_anchor_k1", "front")])
        for index in range(6)
    ]
    report = balance_question_labels(groups)
    stratum = _stratum(report, "cross_view_snapshot_anchor_k1")
    assert stratum["collapsed"] is True
    assert stratum["retained"] == 0
    assert stratum["dropped"] == 6
    assert report["retained_count"] == 0
    assert report["collapsed_capabilities"] == ("cross_view_snapshot_anchor_k1",)


def test_capabilities_balance_independently_of_each_other() -> None:
    """Each imagined yaw is its own stratum, so one skew cannot mask another.

    Balancing the whole reference curve as a single pool would let the yaws
    that happen to answer "visible" pay for the yaws that never do, and the
    rotation named in the question text would go on predicting the label.
    """
    groups = [
        _group(
            f"g{index}",
            [
                ("reference_frame_visibility", "visible" if index < 8 else "not_visible"),
                (
                    "reference_frame_visibility_yaw180",
                    "not_visible" if index < 8 else "visible",
                ),
            ],
        )
        for index in range(10)
    ]
    report = balance_question_labels(groups)
    for capability in ("reference_frame_visibility", "reference_frame_visibility_yaw180"):
        stratum = _stratum(report, capability)
        assert stratum["retained_per_label"] == 2, capability
    retained = [entry["capability"] for entry in report["retained"]]
    assert retained.count("reference_frame_visibility") == 4
    assert retained.count("reference_frame_visibility_yaw180") == 4


def test_unanswerable_questions_never_enter_the_balance() -> None:
    """An abstained or invalid instance carries no label to balance against."""
    groups = [
        _group("g0", [("cross_view_snapshot_closer_k3", "first")]),
        _group("g1", [("cross_view_snapshot_closer_k3", "second")]),
        _group("g2", [("cross_view_snapshot_closer_k3", None)]),
        _group("g3", [("cross_view_snapshot_closer_k3", None)]),
    ]
    report = balance_question_labels(groups)
    stratum = _stratum(report, "cross_view_snapshot_closer_k3")
    assert stratum["label_counts"] == {"first": 1, "second": 1}
    assert report["question_count"] == 2
    assert report["retained_count"] == 2


def test_selection_is_deterministic_and_names_the_groups_it_keeps() -> None:
    """Reruns must ship the same questions so a collection stays reproducible."""
    groups = [
        _group(f"g{index:02d}", [("cross_view_snapshot_ego_k2", label)])
        for index, label in enumerate(["left"] * 3 + ["right"] * 5)
    ]
    first = balance_question_labels(groups)
    second = balance_question_labels(list(reversed(groups)))
    assert first["retained"] == second["retained"]
    assert [entry["question_group_id"] for entry in first["retained"]] == [
        "g00",
        "g01",
        "g02",
        "g03",
        "g04",
        "g05",
    ]

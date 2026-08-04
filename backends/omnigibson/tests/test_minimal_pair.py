from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnigibson_episode.minimal_pair import _visible_views, certify_minimal_pair


def test_visible_views_selects_exact_entity_identity() -> None:
    episode = {
        "observations": [
            {"view_id": "view-000", "visible_entity_ids": ["chair-a", "table-a"]},
            {"view_id": "view-001", "visible_entity_ids": ["table-a"]},
            {"view_id": "view-002", "visible_entity_ids": ["chair-a"]},
        ]
    }

    assert _visible_views(episode, "chair-a") == ["view-000", "view-002"]


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_model_swap_pair_certifies_answer_invariance(tmp_path: Path) -> None:
    base, variant = tmp_path / "base", tmp_path / "variant"
    target = {
        "source_entity_id": "chair_0",
        "entity_id": "chair-id",
        "raw_label": "chair",
        "obb": {"center_m": [-1.5, 0.0, 0.5]},
    }
    anchor = {
        "source_entity_id": "table_0",
        "entity_id": "table-id",
        "raw_label": "table",
        "obb": {"center_m": [0.0, 0.0, 0.5]},
    }
    for root, variant_name in ((base, "canonical"), (variant, "model_swap__p")):
        _write(root / "trajectory_plan.json", {"same": True})
        _write(root / "render_report.json", {"status": "success"})
        _write(root / "scene_ir.json", {"entities": [target, anchor]})
        _write(
            root / "spatial_episode.json",
            {
                "episode_id": f"episode-{variant_name}",
                "scene_id": "scene",
                "family_id": "family",
                "split_group": "scene:one",
                "family_variant": variant_name,
                "observations": [
                    {
                        "view_id": "view-000",
                        "visible_entity_ids": ["chair-id", "table-id"],
                    }
                ],
            },
        )
    _write(
        base / "scene_snapshot.json",
        {
            "entities": [
                {"source_entity_id": "chair_0", "category": "chair", "model": "aaaaaa"}
            ]
        },
    )
    _write(
        variant / "scene_snapshot.json",
        {
            "entities": [
                {"source_entity_id": "chair_0", "category": "chair", "model": "bbbbbb"}
            ]
        },
    )
    _write(
        variant / "intervention_execution.json",
        {
            "status": "success",
            "intervention_type": "single_object_model_swap",
            "proposal": {
                "proposal_id": "p",
                "target_source_entity_id": "chair_0",
                "anchor_source_entity_id": "table_0",
                "target_category": "chair",
                "source_model": "aaaaaa",
                "replacement_model": "bbbbbb",
                "relation_axis": "x",
                "relation_before": "left_of",
                "relation_after": "left_of",
            },
            "checks": [],
        },
    )

    certificate = certify_minimal_pair(base_bundle=base, variant_bundle=variant)

    assert certificate["status"] == "certified"
    assert certificate["learning_signal"] == "answer_invariance"
    assert certificate["model_mutation"]["replacement_model"] == "bbbbbb"

    shifted_anchor = {
        **anchor,
        "obb": {"center_m": [0.03, 0.0, 0.5]},
    }
    _write(variant / "scene_ir.json", {"entities": [target, shifted_anchor]})
    with pytest.raises(ValueError, match="single_mutable_entity"):
        certify_minimal_pair(base_bundle=base, variant_bundle=variant)

    failure = json.loads((variant / "minimal_pair_failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["failed_checks"] == ["single_mutable_entity"]
    assert any(
        item["name"] == "single_mutable_entity" and not item["passed"]
        for item in failure["checks"]
    )

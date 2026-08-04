from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "data" / "transform_pilot_v1" / "preflight"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_preflight_counts_independent_facts_separately_from_language() -> None:
    census = _json(PREFLIGHT / "census.json")
    assert census["schema_version"] == "epispace.transform_preflight.v1"
    assert census["stage"] == "geometry_only_preflight"
    assert census["training_exports_present"] is False
    assert census["counts"]["self_rotation"]["language_surface_variants"] == 0
    assert census["counts"]["among5"]["language_surface_variants"] == 0
    assert census["status"] == "review_required"


def test_self_rotation_target_orientation_never_leaks_into_model_input() -> None:
    config = _json(ROOT / "configs" / "transform_pilot_v1.json")
    records = _jsonl(PREFLIGHT / "self_rotation.base_facts.jsonl")
    assert records
    for row in records:
        target = row["oracle"]["target_orientation_view_id"]
        assert target not in row["model_view_ids"]
        assert row["oracle"]["target_orientation_is_model_visible"] is False
        assert row["answer"]["center_margin_deg"] >= config["direction_contract"][
            "minimum_center_margin_deg"
        ]
        assert row["answer"]["effective_margin_deg"] >= config["direction_contract"][
            "minimum_obb_aware_margin_deg"
        ]


def test_among5_records_have_five_unique_model_facing_objects_and_connected_views() -> None:
    records = _jsonl(PREFLIGHT / "among5.base_layouts.jsonl")
    census = _json(PREFLIGHT / "census.json")
    assert len(records) == (
        census["counts"]["among5"]["primary_orbit_base_layouts"]
        + census["counts"]["among5"]["auxiliary_coverage_base_layouts"]
    )
    for row in records:
        categories = [row["anchor"]["category"], *[item["category"] for item in row["satellites"]]]
        directions = {item["relation_to_anchor"]["label"] for item in row["satellites"]}
        assert len(categories) == len(set(categories)) == 5
        assert directions == {"front", "right", "back", "left"}
        assert len(row["model_view_ids"]) == 4
        assert row["view_registration_graph"]["connected"] is True
        assert row["program_minimality"]["unknown_label_authorized"] is False

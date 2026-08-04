from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web" / "data" / "generalization_episode.v1.json"
PILOT_SFT = ROOT / "web" / "data" / "pilot_qwen_sft.v1.jsonl"


def _episode() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_demo_is_real_and_complete() -> None:
    episode = _episode()
    assert episode["schema_version"] == "episode3d.generalization_episode.v1"
    assert episode["source"]["quality_status"] == "pass"
    assert episode["source"]["loop_rgb_psnr_db"] >= 30.0
    assert len(episode["views"]) == 11
    assert len(episode["queries"]) == 9
    assert len(episode["family_variants"]) == 5


def test_grounding_and_cross_view_certificates_are_executable() -> None:
    queries = {query["query_id"]: query for query in _episode()["queries"]}
    bbox = queries["G-fridge-view000"]["answer"]["bbox_norm_xyxy"]
    assert all(0.0 <= value <= 1.0 for value in bbox)
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]
    assert queries["G-fridge-view000"]["answer"]["visible_pixels"] > 128
    cross_view = queries["CR-tv-vs-fridge-cross-view"]
    assert cross_view["answer"]["co_visible_views"] == []
    assert cross_view["certificate"][0] == {
        "check": "no_common_evidence_view",
        "pass": True,
    }


def test_unknown_and_oracle_leakage_contracts() -> None:
    episode = _episode()
    unknown = next(query for query in episode["queries"] if query["status"] == "unknown")
    assert unknown["answer"]["value"] is None
    assert unknown["answer"]["status"] == "unknown"
    for query in episode["queries"]:
        inputs = set(query["loss_policy"]["input_tokens"])
        assert not inputs & set(episode["channel_policy"]["oracle_only"])
        assert "certificate_expected_answer" in query["loss_policy"]["masked_out"]


def test_family_variants_cannot_cross_splits() -> None:
    episode = _episode()
    assert all(variant["family_lock"] for variant in episode["family_variants"])
    train_atoms = set(episode["research_contract"]["train_atoms"])
    graph_atoms = {
        node["op"]
        for query in episode["queries"]
        for node in query["operation_graph"]
    }
    assert graph_atoms <= train_atoms
    held_out = set(episode["research_contract"]["held_out_signatures"])
    assert held_out <= {query["program_signature"] for query in episode["queries"]}


def test_pilot_uses_observable_anchor_and_safe_rewrite_api() -> None:
    pilot = _episode()["pilot_study"]
    state = pilot["canonical_state_target"]
    assert state["frame_id"] == "episode_map@view-000"
    assert state["frame_contract"]["origin"] == "anchor camera ground projection"
    assert state["entity_count"] == 59
    assert len(state["entities"]) == state["entity_count"]
    assert all("entity_id" not in item for item in state["entities"])
    assert all(
        not example["safe_rewrite_api"]["oracle_answer_visible_to_subagent"]
        for example in pilot["examples"]
    )


def test_pilot_numeric_targets_are_trainable_not_false_precision() -> None:
    examples = {item["query_id"]: item for item in _episode()["pilot_study"]["examples"]}
    grounding = examples["G-fridge-view000"]
    assert grounding["node_targets"]["bbox_0_1000_xyxy"] == [805, 498, 914, 666]
    frame = examples["F-view000-view002"]["node_targets"]
    assert frame["translation_anchor_m"] == [0.408, 1.964]
    assert frame["yaw_clockwise_deg"] == 22.211
    metric = examples["M-coffee-fridge"]
    assert "4.4 米" in metric["answer_surface_zh"]
    assert metric["numeric_policy"]["tolerance"] == "max(0.3m,10%)。"


def test_qwen_record_excludes_held_out_and_oracle_fields() -> None:
    pilot = _episode()["pilot_study"]
    record = pilot["qwen_sft_record"]
    assistant = next(message["content"] for message in record["messages"] if message["role"] == "assistant")
    user = next(message["content"] for message in record["messages"] if message["role"] == "user")
    user_text = " ".join(block.get("text", "") for block in user if block["type"] == "text")
    for query_id in record["exclusions"]["held_out_query_ids"]:
        assert query_id not in assistant
        assert query_id not in user_text
    assert "runtime_instance_id" not in assistant
    assert "entity_uuid" not in assistant
    assert PILOT_SFT.exists()
    assert json.loads(PILOT_SFT.read_text(encoding="utf-8")) == record


def test_visibility_candidate_is_rejected_until_target_view_is_hidden() -> None:
    blocked = _episode()["pilot_study"]["blocked_candidate"]
    assert blocked["query_id"] == "P-fridge-view005"
    assert "target RGB" in blocked["reason"]

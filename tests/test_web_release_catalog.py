from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from scripts.build_web_release_catalog import (
    CATALOG_SCHEMA,
    DETAIL_SCHEMA,
    SEMANTIC_PACKET_PATH,
    SEMANTIC_RESULT_PATH,
    WebReleaseBuildError,
    build_web_release_catalog,
)

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
RELEASE = ROOT / "data" / "epispace_pilot_v1"
CATALOG_PATH = WEB_ROOT / "data" / "release_catalog.v1.json"
DETAILS_DIR = WEB_ROOT / "data" / "release_details"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_catalog_is_bound_to_the_passing_final_release() -> None:
    catalog = _json(CATALOG_PATH)
    final_index = _json(RELEASE / "final_release_index.json")
    manifest = _json(RELEASE / "release_manifest.json")
    assert catalog["schema_version"] == CATALOG_SCHEMA
    assert catalog["status"] == "verified"
    assert final_index["status"] == manifest["status"] == "pass"
    provenance = catalog["provenance"]
    assert provenance["dataset_id"] == manifest["dataset_id"]
    assert provenance["release_manifest_sha256"] == _sha256(
        RELEASE / "release_manifest.json"
    )
    assert provenance["final_release_index_sha256"] == _sha256(
        RELEASE / "final_release_index.json"
    )
    assert provenance["release_manifest_sha256"] == final_index[
        "release_manifest_sha256"
    ]
    assert provenance["episode_ir_sha256"] == final_index["base_artifacts"][
        "episode_ir"
    ]["sha256"]


def test_semantic_visual_quality_is_hash_bound_and_never_called_human() -> None:
    catalog = _json(CATALOG_PATH)
    final_index = _json(RELEASE / "final_release_index.json")
    authority = final_index["semantic_visual_audit"]
    quality = catalog["quality"]["semantic_visual_audit"]
    verification = catalog["verification"]["semantic_visual_audit"]
    result = _json(RELEASE / SEMANTIC_RESULT_PATH)
    packet = _json(RELEASE / SEMANTIC_PACKET_PATH)

    assert authority["status"] == result["status"] == quality["status"] == "pass"
    assert authority["gate"] is result["decision"]["semantic_visual_audit_gate"] is True
    assert quality["gate"] is verification["gate"] is True
    assert quality["packet_id"] == authority["packet"]["packet_id"] == packet["packet_id"]
    assert quality["result_id"] == authority["result"]["result_id"] == result["result_id"]
    assert quality["seed"] == packet["sampling_contract"]["seed"]
    assert quality["sample_size"] == len(packet["items"]) == result["summary"]["reviewed_items"]
    assert quality["counts"] == result["summary"]["overall_status_counts"]
    assert quality["counts"]["major_issue"] == quality["counts"]["unreviewable"] == 0
    assert quality["reviewer_type"] == "model_assisted_independent"
    assert verification["reviewer_type"] == quality["reviewer_type"]
    assert verification["sample_size"] == quality["sample_size"]
    assert verification["counts"] == quality["counts"]
    assert quality["reviewer_label_zh"] == "模型辅助独立复核（非人工审计）"
    assert "human audit" in quality["claim_boundary"]
    assert quality["hash_binding"]["packet_sha256"] == _sha256(
        RELEASE / SEMANTIC_PACKET_PATH
    )
    assert quality["hash_binding"]["result_sha256"] == _sha256(
        RELEASE / SEMANTIC_RESULT_PATH
    )
    assert quality["hash_binding"]["review_evidence_sha256"] == packet[
        "review_evidence_binding"
    ]["sha256"]
    assert sorted(quality["hash_binding"]["review_sha256"]) == sorted(
        item["sha256"] for item in authority["reviews"]
    )


def test_release_counts_are_authoritative_and_exhaustive() -> None:
    catalog = _json(CATALOG_PATH)
    manifest = _json(RELEASE / "release_manifest.json")
    statistics = manifest["statistics"]
    assert catalog["funnel"] == {
        "planned_jobs": 322,
        "strict_passed_bundles": 193,
        "write_source_bundles": 172,
        "oracle_verifier_bundles": 21,
        "compiled_episodes": 172,
        "compiled_views": 1063,
        "compiled_questions": 1093,
        "episode_supervised_facts": 651,
    }
    assert len(catalog["episodes"]) == statistics["compiled"][
        "source_trajectory_bundles"
    ]
    assert catalog["exports"] == statistics["exports"]
    assert catalog["splits"]["scenes"] == statistics["compiled"]["split_scenes"]
    assert catalog["splits"]["episodes"] == statistics["compiled"]["split_bundles"]
    assert sum(item["question_count"] for item in catalog["episodes"]) == 1093
    assert sum(item["view_count"] for item in catalog["episodes"]) == 1063
    assert len({item["episode_id"] for item in catalog["episodes"]}) == 172


def test_every_lazy_detail_closes_over_real_rgb_without_host_paths() -> None:
    catalog = _json(CATALOG_PATH)
    details = list(DETAILS_DIR.glob("*.json"))
    assert len(details) == len(catalog["episodes"]) == 172
    manifest_sha = catalog["provenance"]["release_manifest_sha256"]
    for entry in catalog["episodes"]:
        detail_path = WEB_ROOT / entry["detail_url"]
        assert detail_path.is_file()
        detail = _json(detail_path)
        assert detail["schema_version"] == DETAIL_SCHEMA
        assert detail["identity"]["episode_id"] == entry["episode_id"]
        assert detail["release_binding"]["release_manifest_sha256"] == manifest_sha
        observations = detail["model_visible"]["observations"]
        assert len(observations) == entry["view_count"]
        assert observations[0]["rgb_url"] == entry["thumbnail_url"]
        for observation in observations:
            assert (WEB_ROOT / observation["rgb_url"]).resolve().is_file()
    for value in _walk(catalog):
        if isinstance(value, str):
            assert not Path(value).is_absolute()
            assert "/data/shichao/" not in value
    for path in details:
        for value in _walk(_json(path)):
            if isinstance(value, str):
                assert not Path(value).is_absolute()
                assert "/data/shichao/" not in value


def test_model_visible_surface_has_no_oracle_or_compiler_leakage() -> None:
    forbidden_keys = {
        "answer",
        "answer_zh",
        "answer_value",
        "certificate",
        "depth",
        "instance",
        "oracle",
        "pose",
        "program",
        "semantic",
        "state",
        "target",
    }
    for path in DETAILS_DIR.glob("*.json"):
        detail = _json(path)
        visible = detail["model_visible"]
        assert set(visible) == {"policy", "observations", "questions"}
        assert all(set(item) == {"view_id", "step", "role", "rgb_url"} for item in visible["observations"])
        assert all(set(item) == {"order", "question_zh"} for item in visible["questions"])
        keys = {
            str(key).lower()
            for value in _walk(visible)
            if isinstance(value, dict)
            for key in value
        }
        assert not keys & forbidden_keys
        assert detail["supervision_target"]["answers"]
        assert detail["compiler_proof"]["questions"]
        for proof in detail["compiler_proof"]["questions"]:
            assert proof["certificate"]["result"] in {"pass", "unknown"}
            assert all(check["passed"] is True for check in proof["certificate"]["checks"])


def test_artifact_links_and_hashes_resolve_to_release_files() -> None:
    catalog = _json(CATALOG_PATH)
    manifest = _json(RELEASE / "release_manifest.json")
    assert {item["name"] for item in catalog["artifacts"]} == set(
        manifest["artifacts"]
    )
    for item in catalog["artifacts"]:
        path = (WEB_ROOT / item["url"]).resolve()
        assert path == (RELEASE / manifest["artifacts"][item["name"]]).resolve()
        assert path.is_file()
        assert _sha256(path) == item["sha256"]
        assert path.stat().st_size == item["bytes"]


@pytest.mark.parametrize("failure", ["status", "manifest_hash", "artifact_hash"])
def test_generator_fails_closed_and_removes_stale_web_output(
    tmp_path: Path, failure: str
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    shutil.copy2(RELEASE / "final_release_index.json", release / "final_release_index.json")
    shutil.copy2(RELEASE / "release_manifest.json", release / "release_manifest.json")
    if failure == "status":
        index = _json(release / "final_release_index.json")
        index["status"] = "fail"
        (release / "final_release_index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
    elif failure == "manifest_hash":
        with (release / "release_manifest.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
    else:
        (release / "benchmark.jsonl").write_text("{}\n", encoding="utf-8")
    web_data = tmp_path / "web" / "data"
    output = web_data / "release_catalog.v1.json"
    details = web_data / "release_details"
    details.mkdir(parents=True)
    output.write_text("stale", encoding="utf-8")
    (details / "stale.json").write_text("stale", encoding="utf-8")
    with pytest.raises(WebReleaseBuildError):
        build_web_release_catalog(release, output, details)
    assert not output.exists()
    assert not details.exists()


@pytest.mark.parametrize("failure", ["missing", "stale_hash", "gate_false"])
def test_generator_fails_closed_on_semantic_audit_and_removes_projection(
    tmp_path: Path, failure: str
) -> None:
    release = tmp_path / "release"
    shutil.copytree(RELEASE, release)
    result_path = release / SEMANTIC_RESULT_PATH
    if failure == "missing":
        result_path.unlink()
    elif failure == "stale_hash":
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
    else:
        index = _json(release / "final_release_index.json")
        index["semantic_visual_audit"]["gate"] = False
        (release / "final_release_index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )

    web_data = tmp_path / "web" / "data"
    output = web_data / "release_catalog.v1.json"
    details = web_data / "release_details"
    details.mkdir(parents=True)
    output.write_text("stale", encoding="utf-8")
    (details / "stale.json").write_text("stale", encoding="utf-8")
    with pytest.raises(WebReleaseBuildError, match="semantic"):
        build_web_release_catalog(release, output, details)
    assert not output.exists()
    assert not details.exists()


def test_web_integrates_verified_release_without_old_overclaims() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    assert 'href="#release">数据资产</a>' in html
    assert '<section id="release"' in html
    assert 'fetch("data/release_catalog.v1.json")' in javascript
    assert "231 条真实采集" not in html
    assert "ΔS<sub>1:M</sub>" not in html
    assert "状态、节点、验证与 family consistency 有监督" not in html
    assert "MULTI-TASK OBJECTIVE" not in html
    assert "视觉 token、优化步数和基础模型必须配平" not in html
    assert 'id="release-semantic-title"' in html
    assert "release.quality.semantic_visual_audit" in javascript
    assert "模型辅助独立复核（非人工审计）" not in html

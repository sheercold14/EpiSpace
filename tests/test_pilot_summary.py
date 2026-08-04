from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from episode3d.pilot_summary import (
    HELDOUT_COMPOSITION_SIGNATURES,
    PilotLayout,
    PilotSummaryError,
    render_markdown,
    sha256,
    summarize_pilot,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metric(accuracy: float, total: int = 4) -> dict:
    correct = round(accuracy * total)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "status": "defined",
    }


def _evaluation(benchmark: Path, predictions: Path, accuracy: float) -> dict:
    benchmark_rows = [json.loads(line) for line in benchmark.read_text().splitlines()]
    by_program: dict[str, list[dict]] = {}
    for row in benchmark_rows:
        by_program.setdefault(row["program"]["program_id"], []).append(row)
    correctness: dict[str, bool] = {}
    per_program = {}
    for program_id, rows in sorted(by_program.items()):
        metric = _metric(accuracy, len(rows))
        per_program[program_id] = metric
        for index, row in enumerate(rows):
            correctness[row["record_id"]] = index < metric["correct"]
    record_correct = sum(correctness.values())
    record_total = len(benchmark_rows)
    return {
        "schema_version": "epispace.benchmark_evaluation.v1",
        "benchmark": {"records": record_total, "families": 2, "consistency_groups": 1},
        "predictions": {
            "rows": record_total,
            "missing_count": 0,
            "extra_count": 0,
            "duplicate_id_count": 0,
        },
        "record_accuracy": {
            "correct": record_correct,
            "total": record_total,
            "accuracy": record_correct / record_total,
            "status": "defined",
        },
        "family_exact_match": _metric(accuracy, 4),
        "claim_pair_exact_match": _metric(accuracy, 4),
        "frame_equivariance": {"frame_pair_exact_match": _metric(accuracy, 4)},
        "evidence_triples": {
            "revealed_accuracy": _metric(accuracy, 4),
            "exact_match": _metric(accuracy, 4),
            "accuracy_conditioned_evidence_sensitivity": {
                "numerator_joint_correct": round(accuracy * 4),
                "denominator_revealed_correct": 4,
                "value": accuracy,
                "status": "defined",
            },
        },
        "per_program": per_program,
        "record_results": [
            {"record_id": row["record_id"], "correct": correctness[row["record_id"]]}
            for row in benchmark_rows
        ],
        "inputs": {
            "benchmark": str(benchmark.resolve()),
            "benchmark_sha256": sha256(benchmark),
            "predictions": str(predictions.resolve()),
            "predictions_sha256": sha256(predictions),
        },
    }


def _make_run(
    inference_dir: Path,
    *,
    benchmark: Path,
    model: Path,
    accuracy: float,
    image_mode: str,
    train_dir: Path | None = None,
    schedule: Path | None = None,
) -> None:
    ids = [json.loads(line)["record_id"] for line in benchmark.read_text().splitlines()]
    predictions = inference_dir / "predictions.jsonl"
    raw = inference_dir / "raw_generations.jsonl"
    _write_jsonl(predictions, [{"record_id": record_id, "answer_value": None} for record_id in ids])
    _write_jsonl(raw, [{"record_id": record_id, "model_output": "x"} for record_id in ids])
    adapter = train_dir / "adapter" if train_dir else None
    manifest = {
        "schema_version": "epispace.qwen3vl_predictions.v1",
        "model": str(model.resolve()),
        "adapter": str(adapter.resolve()) if adapter else None,
        "benchmark": str(benchmark.resolve()),
        "benchmark_sha256": sha256(benchmark),
        "records": len(ids),
        "parsed": len(ids),
        "num_shards": 1,
        "image_mode": image_mode,
        "image_min_pixels": 65_536,
        "image_max_pixels": 65_536,
        "max_new_tokens": 64,
        "predictions": str(predictions.resolve()),
        "raw_generations": str(raw.resolve()),
    }
    _write_json(inference_dir / "prediction_manifest.json", manifest)
    evaluation = _evaluation(benchmark, predictions, accuracy)
    _write_json(inference_dir / "evaluation" / "evaluation.json", evaluation)
    (inference_dir / "evaluation" / "evaluation.md").write_text("# evaluation\n", encoding="utf-8")
    if train_dir is not None:
        assert schedule is not None and adapter is not None
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
        _write_json(adapter / "adapter_config.json", {"r": 8})
        _write_jsonl(train_dir / "train_log.jsonl", [{"update_step": 1}])
        _write_json(
            train_dir / "run_manifest.json",
            {
                "schema_version": "epispace.qwen3vl_lora_run.v1",
                "status": "complete",
                "config": {
                    "model": str(model.resolve()),
                    "schedule": str(schedule.resolve()),
                    "seed": 17,
                    "max_steps": None,
                },
                "schedule_sha256": sha256(schedule),
                "draws": 1,
            },
        )


def _fixture(tmp_path: Path) -> PilotLayout:
    root = tmp_path / "experiments" / "epispace_qwen3vl4b"
    benchmark = tmp_path / "benchmark.core.jsonl"
    rows = []
    program_order = [
        "counterfactual_cross_view.v1",
        "target_view_prediction.v1",
        "evidence_presence_unknown.v1",
    ]
    for program_offset, program_id in enumerate(program_order):
        signature = HELDOUT_COMPOSITION_SIGNATURES.get(program_id, "G->V")
        for local_index in range(4):
            index = program_offset * 4 + local_index
            rows.append(
                {
                    "record_id": f"record-{index}",
                    "scene_id": f"scene-{local_index}",
                    "program": {
                        "program_id": program_id,
                        "semantic_signature": signature,
                    },
                }
            )
    _write_jsonl(benchmark, rows)
    model = tmp_path / "model"
    _write_json(model / "config.json", {"model_type": "qwen3_vl"})
    (model / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (model / "model-00002-of-00002.safetensors").write_bytes(b"second")
    _write_json(
        model / "model.safetensors.index.json",
        {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        },
    )
    layout = PilotLayout.defaults(root, benchmark, model)
    _write_jsonl(
        layout.composition_benchmark,
        [row for row in rows if row["program"]["program_id"] in HELDOUT_COMPOSITION_SIGNATURES],
    )
    _write_jsonl(layout.episode_schedule, [{"draw": 0}])
    _write_jsonl(layout.isolated_schedule, [{"draw": 0}])
    _write_json(
        layout.schedule_manifest,
        {
            "status": "pass",
            "artifacts": {
                "episode": {
                    "path": layout.episode_schedule.name,
                    "sha256": sha256(layout.episode_schedule),
                },
                "isolated": {
                    "path": layout.isolated_schedule.name,
                    "sha256": sha256(layout.isolated_schedule),
                },
            },
        },
    )
    _write_json(
        layout.token_profile,
        {
            "status": "pass",
            "model": str(model.resolve()),
            "sources": {
                "episode_schedule_sha256": sha256(layout.episode_schedule),
                "isolated_schedule_sha256": sha256(layout.isolated_schedule),
            },
        },
    )
    _make_run(
        layout.base_full_dir,
        benchmark=benchmark,
        model=model,
        accuracy=0.5,
        image_mode="full",
    )
    _make_run(
        layout.base_no_image_dir,
        benchmark=benchmark,
        model=model,
        accuracy=0.25,
        image_mode="none",
    )
    _make_run(
        layout.episode_train_dir / "inference_core_256",
        benchmark=benchmark,
        model=model,
        accuracy=0.75,
        image_mode="full",
        train_dir=layout.episode_train_dir,
        schedule=layout.episode_schedule,
    )
    _make_run(
        layout.isolated_train_dir / "inference_core_256",
        benchmark=benchmark,
        model=model,
        accuracy=0.5,
        image_mode="full",
        train_dir=layout.isolated_train_dir,
        schedule=layout.isolated_schedule,
    )
    return layout


def test_complete_summary_binds_artifacts_and_computes_deltas(tmp_path: Path) -> None:
    layout = _fixture(tmp_path)
    summary = summarize_pilot(layout)

    assert summary["status"] == "complete"
    assert summary["claim_scope"]["is_main_result"] is False
    assert summary["complete_runs"] == [
        "base_full",
        "base_no_image",
        "episode_sft",
        "isolated_sft",
    ]
    assert summary["deltas"]["episode_minus_isolated"]["overall"]["record_accuracy"] == 0.25
    assert summary["deltas"]["episode_minus_isolated"]["overall"]["heldout_composition"] == 0.25
    assert summary["contract"]["heldout_composition"]["records"] == 8
    assert summary["runs"]["episode_sft"]["metrics"]["heldout_composition"] == {
        "correct": 6,
        "total": 8,
        "accuracy": 0.75,
        "status": "defined",
        "program_ids": [
            "counterfactual_cross_view.v1",
            "target_view_prediction.v1",
        ],
        "preregistered_primary_endpoint": True,
    }
    assert (
        summary["deltas"]["episode_minus_isolated"]["per_program"]["counterfactual_cross_view.v1"]
        == 0.25
    )
    model_inventory = summary["provenance"]["model"]
    assert model_inventory["file_count"] == 4
    assert {entry["role"] for entry in model_inventory["files"]} == {
        "config",
        "weight_index",
        "weight",
    }
    assert summary["runs"]["episode_sft"]["artifacts"]["adapter"]["file_count"] == 2
    uncertainty = summary["sampling_uncertainty"]
    assert uncertainty["status"] == "defined"
    assert uncertainty["record_accuracy"]["resamples"] == 10_000
    assert uncertainty["heldout_composition"]["observed_delta"] == 0.25
    markdown = render_markdown(summary)
    assert "not the multi-seed/8B main result" in markdown
    assert "episode_minus_isolated" in markdown
    assert "sampling uncertainty" in markdown
    assert "Held-out composition" in markdown


def test_missing_run_fails_by_default_and_is_explicit_when_allowed(tmp_path: Path) -> None:
    layout = _fixture(tmp_path)
    shutil.rmtree(layout.isolated_train_dir / "inference_core_256")

    with pytest.raises(PilotSummaryError, match="isolated_sft is incomplete"):
        summarize_pilot(layout)
    summary = summarize_pilot(layout, allow_incomplete=True)
    assert summary["status"] == "incomplete"
    assert summary["missing_runs"] == ["isolated_sft"]
    assert "episode_minus_isolated" not in summary["deltas"]


def test_allow_incomplete_rejects_partially_materialized_inference(tmp_path: Path) -> None:
    layout = _fixture(tmp_path)
    (layout.isolated_train_dir / "inference_core_256" / "evaluation" / "evaluation.json").unlink()

    with pytest.raises(PilotSummaryError, match="partially materialized inference"):
        summarize_pilot(layout, allow_incomplete=True)


def test_allow_incomplete_does_not_mask_stale_evaluation(tmp_path: Path) -> None:
    layout = _fixture(tmp_path)
    path = layout.base_full_dir / "evaluation" / "evaluation.json"
    report = json.loads(path.read_text())
    report["inputs"]["predictions_sha256"] = "0" * 64
    _write_json(path, report)

    with pytest.raises(PilotSummaryError, match="evaluator predictions are stale"):
        summarize_pilot(layout, allow_incomplete=True)


def test_composition_subset_and_signature_are_verified_fail_closed(tmp_path: Path) -> None:
    layout = _fixture(tmp_path)
    rows = [json.loads(line) for line in layout.composition_benchmark.read_text().splitlines()]
    rows[0]["program"]["semantic_signature"] = "shortcut"
    _write_jsonl(layout.composition_benchmark, rows)

    with pytest.raises(PilotSummaryError, match="signature mismatch"):
        summarize_pilot(layout)


def test_evaluator_composition_denominator_must_match_frozen_subset(tmp_path: Path) -> None:
    layout = _fixture(tmp_path)
    path = layout.base_full_dir / "evaluation" / "evaluation.json"
    report = json.loads(path.read_text())
    metric = report["per_program"]["target_view_prediction.v1"]
    metric.update({"correct": 1, "total": 3, "accuracy": 1 / 3})
    _write_json(path, report)

    with pytest.raises(PilotSummaryError, match="does not match composition benchmark"):
        summarize_pilot(layout)


def test_evaluator_program_correctness_must_match_record_results(tmp_path: Path) -> None:
    layout = _fixture(tmp_path)
    path = layout.base_full_dir / "evaluation" / "evaluation.json"
    report = json.loads(path.read_text())
    metric = report["per_program"]["target_view_prediction.v1"]
    metric.update({"correct": 1, "total": 4, "accuracy": 0.25})
    _write_json(path, report)

    with pytest.raises(PilotSummaryError, match="disagrees with record_results"):
        summarize_pilot(layout)

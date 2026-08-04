from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from episode3d.benchmark_evaluator import (
    evaluate_files,
)
from episode3d.benchmark_evaluator import (
    render_markdown as render_evaluation,
)
from episode3d.groupstep_pilot_summary import (
    GroupStepPilotLayout,
    GroupStepSummaryError,
    render_markdown,
    sha256,
    summarize_groupstep_pilot,
)
from episode3d.pilot_summary import (
    HELDOUT_COMPOSITION_SIGNATURES,
    write_inventory_artifact,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _benchmark_row(
    index: int,
    *,
    split: str,
    program: str,
    task: str,
    answer_value,
    family_variant: str = "canonical",
    consistency_group: str | None = None,
    scene: int | None = None,
) -> dict:
    return {
        "record_id": f"benchmark-{index:03d}",
        "schema_version": "epispace.benchmark.v1",
        "scene_id": f"scene-{scene if scene is not None else index % 3}",
        "family_id": f"family-{consistency_group or index}",
        "split": split,
        "family_variant": family_variant,
        "consistency_group": consistency_group,
        "presentation_contract": "numbered_answer_list.v1",
        "target": {"task_type": task, "answer_value": answer_value},
        "program": {
            "program_id": program,
            "semantic_signature": HELDOUT_COMPOSITION_SIGNATURES.get(program, "G->V"),
        },
    }


def _make_benchmark(root: Path) -> tuple[Path, Path, list[dict]]:
    rows: list[dict] = []
    for index in range(8):
        rows.append(
            _benchmark_row(
                index,
                split="test",
                program="counterfactual_cross_view.v1",
                task="counterfactual_verification",
                answer_value={"claim_correct": True, "relation": "left_of"},
                scene=index % 3,
            )
        )
    rows.append(
        _benchmark_row(
            8,
            split="test",
            program="target_view_prediction.v1",
            task="target_view_prediction",
            answer_value=True,
            scene=2,
        )
    )
    evidence_variants = (
        ("prefix_unknown", "evidence_presence_unknown.v1", "evidence_presence_unknown", None),
        (
            "revealed",
            "evidence_presence_reveal.v1",
            "evidence_presence_reveal",
            {"status": "present", "category": "chair"},
        ),
        (
            "decisive_deleted",
            "evidence_presence_unknown.v1",
            "evidence_presence_unknown",
            None,
        ),
    )
    next_index = 9
    for split, group, scene in (
        ("test", "evidence-test", 0),
        ("val", "evidence-val", 1),
    ):
        for variant, program, task, answer in evidence_variants:
            rows.append(
                _benchmark_row(
                    next_index,
                    split=split,
                    program=program,
                    task=task,
                    answer_value=answer,
                    family_variant=variant,
                    consistency_group=group,
                    scene=scene,
                )
            )
            next_index += 1
    for variant, claim_correct in (("claim_true", True), ("claim_false", False)):
        rows.append(
            _benchmark_row(
                next_index,
                split="val",
                program="counterfactual_cross_view.v1",
                task="counterfactual_verification",
                answer_value={"claim_correct": claim_correct, "relation": "right_of"},
                family_variant=variant,
                consistency_group="claim-val",
                scene=2,
            )
        )
        next_index += 1
    benchmark = root / "benchmark.core.jsonl"
    composition = root / "benchmark.composition.jsonl"
    _write_jsonl(benchmark, rows)
    _write_jsonl(
        composition,
        [
            row
            for row in rows
            if row["program"]["program_id"] in HELDOUT_COMPOSITION_SIGNATURES
        ],
    )
    return benchmark, composition, rows


def _training_record(
    record_id: str,
    arm: str,
    comparison_id: str,
    fact_ids: list[str],
    image_path: str,
) -> dict:
    return {
        "record_id": record_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "问题"},
                ],
            },
            {"role": "assistant", "content": "1. 答案"},
        ],
        "comparison_contract": {
            "arm": arm,
            "comparison_id": comparison_id,
            "fact_ids": fact_ids,
            "surface_format": "numbered_answer_list.v1",
        },
    }


def _make_schedules(root: Path, model: Path) -> tuple[Path, Path, Path, Path]:
    schedule_root = root / "schedules"
    episode_source = schedule_root / "episode.jsonl"
    isolated_source = schedule_root / "isolated.jsonl"
    image_ab = str((root / "image-ab.png").resolve())
    image_c = str((root / "image-c.png").resolve())
    episode_records = [
        _training_record(
            "episode-ab", "episode", "comparison-ab", ["fact-a", "fact-b"], image_ab
        ),
        _training_record("episode-c", "episode", "comparison-c", ["fact-c"], image_c),
    ]
    isolated_records = [
        _training_record("isolated-a", "isolated", "comparison-ab", ["fact-a"], image_ab),
        _training_record("isolated-b", "isolated", "comparison-ab", ["fact-b"], image_ab),
        _training_record("isolated-c", "isolated", "comparison-c", ["fact-c"], image_c),
    ]
    _write_jsonl(episode_source, episode_records)
    _write_jsonl(isolated_source, isolated_records)
    episode_source_sha = sha256(episode_source)
    isolated_source_sha = sha256(isolated_source)
    episode_schedule = schedule_root / "image_matched.episode.schedule.jsonl"
    isolated_schedule = schedule_root / "image_matched.isolated.schedule.jsonl"
    episode_rows = [
        {
            "arm": "episode",
            "comparison_id": "comparison-ab",
            "record_id": "episode-ab",
            "fact_ids": ["fact-a", "fact-b"],
            "repeat_index": repeat,
            "sample_weight": 0.5,
            "image_paths": [image_ab],
            "source_jsonl": str(episode_source.resolve()),
            "source_jsonl_sha256": episode_source_sha,
            "source_line": 1,
        }
        for repeat in range(2)
    ]
    episode_rows.append(
        {
            "arm": "episode",
            "comparison_id": "comparison-c",
            "record_id": "episode-c",
            "fact_ids": ["fact-c"],
            "repeat_index": 0,
            "sample_weight": 1.0,
            "image_paths": [image_c],
            "source_jsonl": str(episode_source.resolve()),
            "source_jsonl_sha256": episode_source_sha,
            "source_line": 2,
        }
    )
    isolated_rows = []
    for source_line, fact in ((1, "fact-a"), (2, "fact-b")):
        isolated_rows.append(
            {
                "arm": "isolated",
                "comparison_id": "comparison-ab",
                "record_id": f"isolated-{fact[-1]}",
                "fact_ids": [fact],
                "repeat_index": 0,
                "sample_weight": 1.0,
                "image_paths": [image_ab],
                "source_jsonl": str(isolated_source.resolve()),
                "source_jsonl_sha256": isolated_source_sha,
                "source_line": source_line,
            }
        )
    isolated_rows.append(
        {
            "arm": "isolated",
            "comparison_id": "comparison-c",
            "record_id": "isolated-c",
            "fact_ids": ["fact-c"],
            "repeat_index": 0,
            "sample_weight": 1.0,
            "image_paths": [image_c],
            "source_jsonl": str(isolated_source.resolve()),
            "source_jsonl_sha256": isolated_source_sha,
            "source_line": 3,
        }
    )
    _write_jsonl(episode_schedule, episode_rows)
    _write_jsonl(isolated_schedule, isolated_rows)
    compute_manifest = schedule_root / "compute_matching_manifest.json"
    _write_json(
        compute_manifest,
        {
            "status": "pass",
            "sources": {
                "episode": {
                    "path": str(episode_source.resolve()),
                    "sha256": sha256(episode_source),
                }
            },
            "regimes": {
                "image_occurrence_matched": {
                    "schedule_files": {
                        "episode": {
                            "filename": episode_schedule.name,
                            "sha256": sha256(episode_schedule),
                        },
                        "isolated": {
                            "filename": isolated_schedule.name,
                            "sha256": sha256(isolated_schedule),
                        },
                    }
                }
            },
        },
    )
    release_manifest = root / "release_manifest.json"
    _write_json(release_manifest, {"status": "pass"})
    final_release_index = root / "final_release_index.json"
    _write_json(
        final_release_index,
        {
            "status": "pass",
            "release_manifest_sha256": sha256(release_manifest),
            "compute_matching": {
                "manifest_sha256": sha256(compute_manifest),
                "schedules": {
                    "image_occurrence_matched.episode": sha256(episode_schedule),
                    "image_occurrence_matched.isolated": sha256(isolated_schedule),
                },
            },
        },
    )

    def lineage_binding(path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    manifest = schedule_root / "manifest.json"
    _write_json(
        manifest,
        {
            "status": "pass",
            "sources": {
                "episode_sft": sha256(episode_source),
                "episode_schedule": sha256(episode_schedule),
                "isolated_schedule": sha256(isolated_schedule),
            },
            "lineage": {
                "episode_selection_sft": lineage_binding(episode_source),
                "episode_source_schedule": lineage_binding(episode_schedule),
                "isolated_source_schedule": lineage_binding(isolated_schedule),
                "compute_matching_manifest": lineage_binding(compute_manifest),
                "release_manifest": lineage_binding(release_manifest),
                "final_release_index": lineage_binding(final_release_index),
            },
            "lineage_checks": {"formal_release_authority_bound": True},
            "artifacts": {
                "episode": {
                    "path": episode_schedule.name,
                    "sha256": sha256(episode_schedule),
                },
                "isolated": {
                    "path": isolated_schedule.name,
                    "sha256": sha256(isolated_schedule),
                },
            },
        },
    )
    token_profile = schedule_root / "token_profile.json"
    _write_json(
        token_profile,
        {
            "status": "pass",
            "model": str(model.resolve()),
            "image_min_pixels": 65_536,
            "image_max_pixels": 65_536,
            "sources": {
                "episode_schedule_sha256": sha256(episode_schedule),
                "isolated_schedule_sha256": sha256(isolated_schedule),
            },
            "episode": {
                "input_tokens": 120,
                "text_tokens": 100,
                "image_tokens": 20,
                "effective_facts": 3,
            },
            "isolated": {
                "input_tokens": 110,
                "text_tokens": 90,
                "image_tokens": 20,
                "effective_facts": 3,
            },
            "checks": {
                "image_tokens_equal": True,
                "effective_facts_equal": True,
                "effective_fact_weights_one": True,
            },
        },
    )
    return episode_schedule, isolated_schedule, manifest, token_profile


TRAIN_PROTOCOL = {
    "epochs": 2,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "image_min_pixels": 65_536,
    "image_max_pixels": 65_536,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "gradient_checkpointing": False,
    "max_grad_norm": 1.0,
}


def _make_training_run(
    train_dir: Path,
    *,
    model: Path,
    model_inventory: Path,
    schedule: Path,
    paired_schedule: Path,
    seed: int,
) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "adapter").mkdir()
    (train_dir / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    _write_json(train_dir / "adapter" / "adapter_config.json", {"r": 8})
    base_model_provenance = write_inventory_artifact(
        model, model_inventory, kind="model"
    )
    adapter_provenance = write_inventory_artifact(
        train_dir / "adapter",
        train_dir / "adapter_inventory.json",
        kind="directory",
    )
    _write_jsonl(train_dir / "train_log.jsonl", [{"update_step": 4}])
    config = {
        "model": str(model.resolve()),
        "model_inventory": str(model_inventory.resolve()),
        "schedule": str(schedule.resolve()),
        "paired_schedule": str(paired_schedule.resolve()),
        "output_dir": str(train_dir.resolve()),
        "optimizer_unit": "comparison_group",
        "seed": seed,
        **TRAIN_PROTOCOL,
        "gradient_accumulation_steps": 1,
        "max_steps": None,
    }
    _write_json(
        train_dir / "run_manifest.json",
        {
            "schema_version": "epispace.qwen3vl_lora_run.v2",
            "status": "complete",
            "config": config,
            "base_model_inventory_sha256": base_model_provenance[
                "inventory_sha256"
            ],
            "adapter_inventory_sha256": adapter_provenance[
                "inventory_sha256"
            ],
            "base_model_provenance": base_model_provenance,
            "adapter_provenance": adapter_provenance,
            "schedule_sha256": sha256(schedule),
            "paired_schedule_sha256": sha256(paired_schedule),
            "draws": 3,
            "optimizer_unit": "comparison_group",
            "comparison_groups": 2,
            "groups_per_epoch": 2,
            "paired_group_validation": {
                "comparison_groups": 2,
                "draws_per_arm": 3,
                "facts": 3,
            },
            "world_size": 1,
            "local_draws_per_epoch": 3,
            "planned_optimizer_updates": 4,
            "optimizer_updates": 4,
            "completed_comparison_groups": 4,
            "completed_micro_steps_per_rank": 6,
            "loss_contract": (
                "mean CE within each fact; accumulate a comparison group before exactly "
                "one optimizer step"
            ),
        },
    )


def _wrong_answer(expected):
    if isinstance(expected, bool):
        return not expected
    return {"status": "invalid"}


def _make_inference(
    inference_dir: Path,
    *,
    benchmark: Path,
    rows: list[dict],
    model: Path,
    model_inventory: Path,
    adapter: Path | None,
    adapter_inventory: Path | None,
    image_mode: str,
    correctness_offset: int,
) -> None:
    base_model_provenance = write_inventory_artifact(
        model, model_inventory, kind="model"
    )
    adapter_provenance = (
        write_inventory_artifact(
            adapter, adapter_inventory, kind="directory"
        )
        if adapter is not None and adapter_inventory is not None
        else None
    )
    predictions = []
    raw = []
    for index, row in enumerate(rows):
        expected = row["target"]["answer_value"]
        correct = (index + correctness_offset) % 4 != 0
        predictions.append(
            {
                "record_id": row["record_id"],
                "answer_value": expected if correct else _wrong_answer(expected),
                "parse_status": "parsed" if index != len(rows) - 1 else "invalid",
            }
        )
        raw.append({"record_id": row["record_id"], "model_output": "answer"})
    predictions_path = inference_dir / "predictions.jsonl"
    raw_path = inference_dir / "raw_generations.jsonl"
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(raw_path, raw)
    evaluation = evaluate_files(benchmark, predictions_path)
    _write_json(inference_dir / "evaluation" / "evaluation.json", evaluation)
    (inference_dir / "evaluation" / "evaluation.md").write_text(
        render_evaluation(evaluation), encoding="utf-8"
    )
    _write_json(
        inference_dir / "prediction_manifest.json",
        {
            "schema_version": "epispace.qwen3vl_predictions.v1",
            "model": str(model.resolve()),
            "adapter": str(adapter.resolve()) if adapter else None,
            "base_model_inventory_sha256": base_model_provenance[
                "inventory_sha256"
            ],
            "adapter_inventory_sha256": (
                adapter_provenance["inventory_sha256"]
                if adapter_provenance is not None
                else None
            ),
            "base_model_provenance": base_model_provenance,
            "adapter_provenance": adapter_provenance,
            "benchmark": str(benchmark.resolve()),
            "benchmark_sha256": sha256(benchmark),
            "num_shards": 1,
            "records": len(rows),
            "parsed": sum(row["parse_status"] == "parsed" for row in predictions),
            "image_mode": image_mode,
            "prompt_mode": "native",
            "image_min_pixels": 65_536,
            "image_max_pixels": 65_536,
            "max_new_tokens": 64,
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256(predictions_path),
            "raw_generations": str(raw_path.resolve()),
            "raw_generations_sha256": sha256(raw_path),
        },
    )


@dataclass(frozen=True)
class Fixture:
    config: Path
    episode_train_17: Path
    episode_inference_17: Path
    episode_source: Path


def _fixture(tmp_path: Path) -> Fixture:
    model = tmp_path / "model"
    _write_json(model / "config.json", {"model_type": "qwen3_vl"})
    (model / "model.safetensors").write_bytes(b"weights")
    model_inventory = tmp_path / "model_inventory.json"
    write_inventory_artifact(model, model_inventory, kind="model")
    benchmark, composition, rows = _make_benchmark(tmp_path)
    episode_schedule, isolated_schedule, schedule_manifest, token_profile = (
        _make_schedules(tmp_path, model)
    )
    seed_entries = []
    first_episode_train = None
    first_episode_inference = None
    for seed_index, seed in enumerate((17, 23)):
        arm_entries = {}
        for arm_index, (arm, schedule, paired) in enumerate(
            (
                ("episode", episode_schedule, isolated_schedule),
                ("isolated", isolated_schedule, episode_schedule),
            )
        ):
            train_dir = tmp_path / f"seed{seed}-{arm}"
            inference_dir = train_dir / "inference"
            _make_training_run(
                train_dir,
                model=model,
                model_inventory=model_inventory,
                schedule=schedule,
                paired_schedule=paired,
                seed=seed,
            )
            _make_inference(
                inference_dir,
                benchmark=benchmark,
                rows=rows,
                model=model,
                model_inventory=model_inventory,
                adapter=train_dir / "adapter",
                adapter_inventory=train_dir / "adapter_inventory.json",
                image_mode="full",
                correctness_offset=seed_index + arm_index,
            )
            arm_entries[arm] = {
                "train_dir": str(train_dir),
                "inference_dir": str(inference_dir),
            }
            if seed == 17 and arm == "episode":
                first_episode_train = train_dir
                first_episode_inference = inference_dir
        seed_entries.append({"seed": seed, **arm_entries})
    baselines = []
    for name, image_mode, offset in (
        ("base_full", "full", 2),
        ("base_no_image", "none", 3),
    ):
        inference_dir = tmp_path / name
        _make_inference(
            inference_dir,
            benchmark=benchmark,
            rows=rows,
            model=model,
            model_inventory=model_inventory,
            adapter=None,
            adapter_inventory=None,
            image_mode=image_mode,
            correctness_offset=offset,
        )
        baselines.append(
            {"name": name, "inference_dir": str(inference_dir), "image_mode": image_mode}
        )
    config = tmp_path / "groupstep_config.json"
    _write_json(
        config,
        {
            "schema_version": "epispace.groupstep_pilot_config.v1",
            "benchmark": str(benchmark),
            "expected_benchmark_sha256": sha256(benchmark),
            "composition_benchmark": str(composition),
            "model": str(model),
            "schedule_manifest": str(schedule_manifest),
            "token_profile": str(token_profile),
            "episode_schedule": str(episode_schedule),
            "isolated_schedule": str(isolated_schedule),
            "expected_prompt_mode": "native",
            "expected_presentation_contract": "numbered_answer_list.v1",
            "expected_primary_composition_program_counts": {
                "counterfactual_cross_view.v1": 8,
                "target_view_prediction.v1": 1,
            },
            "expected_training_protocol": TRAIN_PROTOCOL,
            "bootstrap_seed": 1701,
            "bootstrap_resamples": 1000,
            "seeds": seed_entries,
            "baselines": baselines,
        },
    )
    assert first_episode_train is not None and first_episode_inference is not None
    return Fixture(
        config=config,
        episode_train_17=first_episode_train,
        episode_inference_17=first_episode_inference,
        episode_source=tmp_path / "schedules" / "episode.jsonl",
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_formal_multiseed_groupstep_summary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary = summarize_groupstep_pilot(GroupStepPilotLayout.from_config(fixture.config))

    assert summary["artifact_status"] == {
        "status": "complete",
        "validated_seed_pairs": 2,
        "validated_baselines": 2,
    }
    assert summary["causal_validity"]["valid"] is True
    assert summary["causal_validity"]["optimizer_unit"] == "comparison_group"
    assert summary["contract"]["seeds"] == [17, 23]
    assert summary["contract"]["primary_endpoint"]["records"] == 9
    assert summary["contract"]["primary_endpoint"]["program_counts"] == {
        "counterfactual_cross_view.v1": 8,
        "target_view_prediction.v1": 1,
    }
    assert summary["matching_scope"]["comparison_groups"] == 2
    assert summary["matching_scope"]["facts"] == 3
    assert summary["matching_scope"]["image_tokens"] == {
        "episode": 20,
        "isolated": 20,
    }
    assert "total input tokens" in summary["matching_scope"]["not_claimed_matched"]
    assert set(summary["seed_runs"]) == {"17", "23"}
    model_digest = summary["provenance"]["model"]["inventory_sha256"]
    seed_run = summary["seed_runs"]["17"]["episode"]
    assert (
        seed_run["training"]["weight_provenance"]["base_model"][
            "inventory_sha256"
        ]
        == model_digest
    )
    assert (
        seed_run["weight_provenance"]["base_model"]["inventory_sha256"]
        == model_digest
    )
    assert (
        seed_run["training"]["weight_provenance"]["adapter"][
            "inventory_sha256"
        ]
        == seed_run["weight_provenance"]["adapter"]["inventory_sha256"]
    )
    assert summary["seed_runs"]["17"]["scene_bootstrap"][
        "test_heldout_composition"
    ]["records"] == 9
    assert summary["seed_runs"]["17"]["episode"]["metrics_by_split"]["test"][
        "frame_pair_exact_match"
    ]["total"] == 0
    assert summary["cross_seed"]["episode"]["test_heldout_composition"][
        "defined_seeds"
    ] == 2
    markdown = render_markdown(summary)
    assert "test-only held-out composition" in markdown
    assert "predictions are not pooled" in markdown
    assert "FLOPs" in markdown


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", "epispace.qwen3vl_lora_run.v1", "not a v2"),
        ("optimizer_unit", "draw", "comparison-group"),
        ("world_size", 2, "world_size"),
        ("optimizer_updates", 3, "groups x epochs"),
        ("completed_micro_steps_per_rank", 5, "draws x epochs"),
        ("paired_schedule_sha256", "0" * 64, "paired schedule hash"),
        ("base_model_inventory_sha256", "0" * 64, "base model inventory digest"),
        ("adapter_inventory_sha256", "0" * 64, "adapter inventory digest"),
    ],
)
def test_groupstep_manifest_tampering_fails_closed(
    tmp_path: Path, field: str, value, error: str
) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = fixture.episode_train_17 / "run_manifest.json"
    manifest = _load(manifest_path)
    manifest[field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(GroupStepSummaryError, match=error):
        summarize_groupstep_pilot(GroupStepPilotLayout.from_config(fixture.config))


def test_gradient_accumulation_and_max_steps_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = fixture.episode_train_17 / "run_manifest.json"
    manifest = _load(manifest_path)
    manifest["config"]["gradient_accumulation_steps"] = 2
    _write_json(manifest_path, manifest)
    with pytest.raises(GroupStepSummaryError, match="gradient accumulation"):
        summarize_groupstep_pilot(GroupStepPilotLayout.from_config(fixture.config))

    manifest["config"]["gradient_accumulation_steps"] = 1
    manifest["config"]["max_steps"] = 1
    _write_json(manifest_path, manifest)
    with pytest.raises(GroupStepSummaryError, match="truncated"):
        summarize_groupstep_pilot(GroupStepPilotLayout.from_config(fixture.config))


def test_prompt_and_surface_mismatches_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prediction_manifest_path = fixture.episode_inference_17 / "prediction_manifest.json"
    prediction_manifest = _load(prediction_manifest_path)
    prediction_manifest["prompt_mode"] = "episode_single"
    _write_json(prediction_manifest_path, prediction_manifest)
    with pytest.raises(GroupStepSummaryError, match="prompt mode"):
        summarize_groupstep_pilot(GroupStepPilotLayout.from_config(fixture.config))

    prediction_manifest["prompt_mode"] = "native"
    _write_json(prediction_manifest_path, prediction_manifest)
    source_rows = [json.loads(line) for line in fixture.episode_source.read_text().splitlines()]
    source_rows[0]["comparison_contract"]["surface_format"] = "legacy_bare_answer.v1"
    _write_jsonl(fixture.episode_source, source_rows)
    config = _load(fixture.config)
    episode_schedule_path = Path(config["episode_schedule"])
    schedule_rows = [
        json.loads(line) for line in episode_schedule_path.read_text().splitlines()
    ]
    source_sha = sha256(fixture.episode_source)
    for row in schedule_rows:
        row["source_jsonl_sha256"] = source_sha
    _write_jsonl(episode_schedule_path, schedule_rows)
    schedule_manifest_path = Path(config["schedule_manifest"])
    schedule_manifest = _load(schedule_manifest_path)
    schedule_manifest["artifacts"]["episode"]["sha256"] = sha256(
        episode_schedule_path
    )
    _write_json(schedule_manifest_path, schedule_manifest)
    token_profile_path = Path(config["token_profile"])
    token_profile = _load(token_profile_path)
    token_profile["sources"]["episode_schedule_sha256"] = sha256(
        episode_schedule_path
    )
    _write_json(token_profile_path, token_profile)
    with pytest.raises(GroupStepSummaryError, match="surface"):
        summarize_groupstep_pilot(GroupStepPilotLayout.from_config(fixture.config))


def test_evaluator_v1_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evaluation_path = fixture.episode_inference_17 / "evaluation" / "evaluation.json"
    evaluation = _load(evaluation_path)
    evaluation["schema_version"] = "epispace.benchmark_evaluation.v1"
    _write_json(evaluation_path, evaluation)

    with pytest.raises(GroupStepSummaryError, match="unsupported evaluator schema"):
        summarize_groupstep_pilot(GroupStepPilotLayout.from_config(fixture.config))


def test_token_profile_resolution_must_match_frozen_training_protocol(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    config = _load(fixture.config)
    token_profile_path = Path(config["token_profile"])
    token_profile = _load(token_profile_path)
    token_profile["image_max_pixels"] = 262_144
    _write_json(token_profile_path, token_profile)

    with pytest.raises(GroupStepSummaryError, match="image_max_pixels"):
        summarize_groupstep_pilot(GroupStepPilotLayout.from_config(fixture.config))


def test_pilot_parent_release_lineage_is_content_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config = _load(fixture.config)
    schedule_manifest_path = Path(config["schedule_manifest"])
    schedule_manifest = _load(schedule_manifest_path)
    final_index_path = Path(
        schedule_manifest["lineage"]["final_release_index"]["path"]
    )
    final_index = _load(final_index_path)
    final_index["status"] = "fail"
    _write_json(final_index_path, final_index)

    with pytest.raises(GroupStepSummaryError, match="final_release_index is stale"):
        summarize_groupstep_pilot(GroupStepPilotLayout.from_config(fixture.config))

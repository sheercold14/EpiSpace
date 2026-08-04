"""Fail-closed reporting for paired, group-step Qwen-VL pilots.

This module is intentionally separate from :mod:`episode3d.pilot_summary`,
which preserves the first per-draw diagnostic.  A report produced here is
valid only after every training run proves that one ``comparison_id`` caused
exactly one Adam update per epoch.  The reporter recomputes the paired
schedule groups instead of trusting the training manifests alone.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from episode3d.pilot_summary import (
    PilotSummaryError,
    directory_inventory,
    extract_metrics,
    model_inventory,
    sha256,
    validate_composition_benchmark,
    validate_inventory_binding,
)
from episode3d.qwen_training import (
    ComparisonGroup,
    ScheduledRecord,
    TrainingContractError,
    build_comparison_groups,
    load_scheduled_records,
    validate_paired_comparison_groups,
)

CONFIG_SCHEMA_VERSION = "epispace.groupstep_pilot_config.v1"
SUMMARY_SCHEMA_VERSION = "epispace.qwen3vl_groupstep_pilot_summary.v1"
TRAIN_MANIFEST_SCHEMA_VERSION = "epispace.qwen3vl_lora_run.v2"
PREDICTION_MANIFEST_SCHEMA_VERSION = "epispace.qwen3vl_predictions.v1"
EVALUATION_SCHEMA_VERSION = "epispace.benchmark_evaluation.v2"

PRIMARY_COMPOSITION_SPLIT = "test"
SIBLING_VARIANTS = {
    "claim_pair": frozenset({"claim_true", "claim_false"}),
    "evidence_triple": frozenset(
        {"prefix_unknown", "revealed", "decisive_deleted"}
    ),
    "frame_pair": frozenset({"frame_a", "frame_b"}),
}

AGGREGATE_METRICS = (
    "record_accuracy",
    "family_exact_match",
    "claim_pair_exact_match",
    "frame_pair_exact_match",
    "revealed_accuracy",
    "evidence_triple_exact_match",
    "aces",
    "test_record_accuracy",
    "test_heldout_composition",
    "test_family_exact_match",
    "test_claim_pair_exact_match",
    "test_frame_pair_exact_match",
    "test_revealed_accuracy",
    "test_evidence_triple_exact_match",
    "test_aces",
    "val_record_accuracy",
    "val_heldout_composition",
    "val_family_exact_match",
    "val_claim_pair_exact_match",
    "val_frame_pair_exact_match",
    "val_revealed_accuracy",
    "val_evidence_triple_exact_match",
    "val_aces",
)
TRAIN_PROTOCOL_KEYS = (
    "model",
    "epochs",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "image_min_pixels",
    "image_max_pixels",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "gradient_checkpointing",
    "max_grad_norm",
)
INFERENCE_PROTOCOL_KEYS = (
    "prompt_mode",
    "image_min_pixels",
    "image_max_pixels",
    "max_new_tokens",
)


class GroupStepSummaryError(RuntimeError):
    """Raised when a formal group-step result cannot be proven valid."""


@dataclass(frozen=True)
class ArmLayout:
    """Training and inference artifacts for one experimental arm."""

    train_dir: Path
    inference_dir: Path


@dataclass(frozen=True)
class SeedLayout:
    """One paired episode/isolated training seed."""

    seed: int
    episode: ArmLayout
    isolated: ArmLayout


@dataclass(frozen=True)
class BaselineLayout:
    """An optional adapter-free baseline inference."""

    name: str
    inference_dir: Path
    image_mode: str


@dataclass(frozen=True)
class GroupStepPilotLayout:
    """All explicit inputs needed for a formal multi-seed pilot report."""

    config_path: Path
    benchmark: Path
    expected_benchmark_sha256: str
    composition_benchmark: Path
    model: Path
    schedule_manifest: Path
    token_profile: Path
    episode_schedule: Path
    isolated_schedule: Path
    expected_prompt_mode: str
    expected_presentation_contract: str
    expected_primary_composition_program_counts: dict[str, int]
    expected_training_protocol: dict[str, Any]
    seeds: tuple[SeedLayout, ...]
    baselines: tuple[BaselineLayout, ...]
    bootstrap_seed: int
    bootstrap_resamples: int

    @classmethod
    def from_config(cls, config_path: Path) -> GroupStepPilotLayout:
        path = config_path.resolve()
        config = _load_json(path)
        _expect(
            config.get("schema_version") == CONFIG_SCHEMA_VERSION,
            f"unsupported group-step config schema in {path}",
        )
        root = path.parent

        def resolve_field(name: str) -> Path:
            value = config.get(name)
            _expect(isinstance(value, str) and value, f"config field {name} is required")
            candidate = Path(value)
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        prompt_mode = config.get("expected_prompt_mode")
        surface = config.get("expected_presentation_contract")
        expected_benchmark_sha256 = config.get("expected_benchmark_sha256")
        expected_primary_counts = config.get(
            "expected_primary_composition_program_counts"
        )
        expected_training_protocol = config.get("expected_training_protocol")
        _expect(
            isinstance(prompt_mode, str) and prompt_mode,
            "expected_prompt_mode must be explicit",
        )
        _expect(
            isinstance(surface, str) and surface,
            "expected_presentation_contract must be explicit",
        )
        _expect(
            isinstance(expected_benchmark_sha256, str)
            and len(expected_benchmark_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_benchmark_sha256),
            "expected_benchmark_sha256 must be an explicit lowercase SHA-256",
        )
        _expect(
            isinstance(expected_training_protocol, dict),
            "expected_training_protocol must be explicit",
        )
        _expect(
            isinstance(expected_primary_counts, dict) and bool(expected_primary_counts),
            "expected_primary_composition_program_counts must be explicit",
        )
        _expect(
            all(
                isinstance(program, str)
                and bool(program)
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
                for program, count in expected_primary_counts.items()
            ),
            "primary composition program counts must be positive integers",
        )
        required_training_keys = set(TRAIN_PROTOCOL_KEYS) - {"model"}
        _expect(
            set(expected_training_protocol) == required_training_keys,
            "expected_training_protocol keys differ from the formal training contract",
        )
        raw_seeds = config.get("seeds")
        _expect(isinstance(raw_seeds, list) and raw_seeds, "config must contain seed pairs")
        seed_layouts: list[SeedLayout] = []
        observed_seeds: set[int] = set()
        for index, raw_seed in enumerate(raw_seeds):
            _expect(isinstance(raw_seed, dict), f"seeds[{index}] is not an object")
            seed = raw_seed.get("seed")
            _expect(
                isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
                f"seeds[{index}].seed is invalid",
            )
            _expect(seed not in observed_seeds, f"duplicate training seed {seed}")
            observed_seeds.add(seed)

            def arm_layout(
                arm: str,
                seed_config: dict[str, Any] = raw_seed,
                seed_index: int = index,
            ) -> ArmLayout:
                value = seed_config.get(arm)
                _expect(
                    isinstance(value, dict),
                    f"seeds[{seed_index}].{arm} is required",
                )
                train_dir = _resolve_from(root, value.get("train_dir"), f"{arm}.train_dir")
                inference_dir = _resolve_from(
                    root, value.get("inference_dir"), f"{arm}.inference_dir"
                )
                return ArmLayout(train_dir=train_dir, inference_dir=inference_dir)

            seed_layouts.append(
                SeedLayout(
                    seed=seed,
                    episode=arm_layout("episode"),
                    isolated=arm_layout("isolated"),
                )
            )

        raw_baselines = config.get("baselines", [])
        _expect(isinstance(raw_baselines, list), "baselines must be a list")
        baselines: list[BaselineLayout] = []
        baseline_names: set[str] = set()
        for index, raw_baseline in enumerate(raw_baselines):
            _expect(isinstance(raw_baseline, dict), f"baselines[{index}] is not an object")
            name = raw_baseline.get("name")
            image_mode = raw_baseline.get("image_mode")
            _expect(isinstance(name, str) and name, f"baselines[{index}].name is invalid")
            _expect(name not in baseline_names, f"duplicate baseline name {name}")
            _expect(
                image_mode in {"full", "none"},
                f"baselines[{index}].image_mode must be full or none",
            )
            baseline_names.add(name)
            baselines.append(
                BaselineLayout(
                    name=name,
                    inference_dir=_resolve_from(
                        root,
                        raw_baseline.get("inference_dir"),
                        f"baselines[{index}].inference_dir",
                    ),
                    image_mode=str(image_mode),
                )
            )

        bootstrap_seed = config.get("bootstrap_seed", 1701)
        bootstrap_resamples = config.get("bootstrap_resamples", 10_000)
        _expect(
            isinstance(bootstrap_seed, int) and not isinstance(bootstrap_seed, bool),
            "bootstrap_seed must be an integer",
        )
        _expect(
            isinstance(bootstrap_resamples, int)
            and not isinstance(bootstrap_resamples, bool)
            and bootstrap_resamples >= 1_000,
            "bootstrap_resamples must be at least 1000",
        )
        return cls(
            config_path=path,
            benchmark=resolve_field("benchmark"),
            expected_benchmark_sha256=expected_benchmark_sha256,
            composition_benchmark=resolve_field("composition_benchmark"),
            model=resolve_field("model"),
            schedule_manifest=resolve_field("schedule_manifest"),
            token_profile=resolve_field("token_profile"),
            episode_schedule=resolve_field("episode_schedule"),
            isolated_schedule=resolve_field("isolated_schedule"),
            expected_prompt_mode=prompt_mode,
            expected_presentation_contract=surface,
            expected_primary_composition_program_counts=dict(
                sorted(expected_primary_counts.items())
            ),
            expected_training_protocol=dict(expected_training_protocol),
            seeds=tuple(sorted(seed_layouts, key=lambda item: item.seed)),
            baselines=tuple(baselines),
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        )


def _resolve_from(root: Path, value: Any, label: str) -> Path:
    _expect(isinstance(value, str) and value, f"{label} must be a path string")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise GroupStepSummaryError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GroupStepSummaryError(f"cannot read valid JSON object from {path}: {exc}") from exc
    _expect(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GroupStepSummaryError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GroupStepSummaryError(f"invalid JSON at {path}:{line_number}") from exc
        _expect(isinstance(value, dict), f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _artifact(path: Path) -> dict[str, Any]:
    _expect(path.is_file(), f"required file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _surface_contract(items: list[ScheduledRecord], arm: str) -> str:
    surfaces: set[str] = set()
    for item in items:
        contract = item.record.get("comparison_contract")
        _expect(isinstance(contract, dict), f"{arm} source record has no comparison contract")
        surface = contract.get("surface_format")
        _expect(isinstance(surface, str) and surface, f"{arm} source has no surface_format")
        surfaces.add(surface)
    _expect(len(surfaces) == 1, f"{arm} schedule mixes presentation surfaces: {surfaces}")
    return next(iter(surfaces))


def _validate_pilot_release_lineage(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Replay the pilot subset's chain back to the final release authority."""

    raw_lineage = manifest.get("lineage")
    _expect(isinstance(raw_lineage, dict), "schedule manifest has no release lineage")
    required = {
        "episode_selection_sft",
        "episode_source_schedule",
        "isolated_source_schedule",
        "compute_matching_manifest",
        "release_manifest",
        "final_release_index",
    }
    _expect(set(raw_lineage) == required, "schedule manifest release lineage is incomplete")
    verified: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        raw = raw_lineage.get(name)
        _expect(isinstance(raw, dict), f"lineage binding {name} is not an object")
        path = (manifest_path.parent / str(raw.get("path", ""))).resolve()
        artifact = _artifact(path)
        _expect(raw.get("sha256") == artifact["sha256"], f"lineage {name} is stale")
        _expect(raw.get("bytes") == artifact["bytes"], f"lineage {name} byte count is stale")
        verified[name] = artifact

    raw_checks = manifest.get("lineage_checks")
    _expect(
        isinstance(raw_checks, dict)
        and bool(raw_checks)
        and all(value is True for value in raw_checks.values()),
        "schedule manifest lineage checks did not all pass",
    )
    sources = manifest.get("sources")
    _expect(isinstance(sources, dict), "schedule manifest has no source bindings")
    _expect(
        sources.get("episode_sft") == verified["episode_selection_sft"]["sha256"],
        "pilot episode selection source is stale",
    )
    _expect(
        sources.get("episode_schedule")
        == verified["episode_source_schedule"]["sha256"],
        "pilot parent episode schedule is stale",
    )
    _expect(
        sources.get("isolated_schedule")
        == verified["isolated_source_schedule"]["sha256"],
        "pilot parent isolated schedule is stale",
    )

    release_manifest = _load_json(Path(verified["release_manifest"]["path"]))
    compute_manifest = _load_json(Path(verified["compute_matching_manifest"]["path"]))
    final_index = _load_json(Path(verified["final_release_index"]["path"]))
    _expect(release_manifest.get("status") == "pass", "parent release manifest did not pass")
    _expect(compute_manifest.get("status") == "pass", "parent compute manifest did not pass")
    _expect(final_index.get("status") == "pass", "parent final release index did not pass")
    _expect(
        final_index.get("release_manifest_sha256")
        == verified["release_manifest"]["sha256"],
        "final release index does not bind the parent manifest",
    )
    final_compute = final_index.get("compute_matching")
    _expect(isinstance(final_compute, dict), "final release index has no compute authority")
    _expect(
        final_compute.get("manifest_sha256")
        == verified["compute_matching_manifest"]["sha256"],
        "final release index does not bind the compute manifest",
    )
    final_schedules = final_compute.get("schedules")
    _expect(isinstance(final_schedules, dict), "final release index has no schedules")
    _expect(
        final_schedules.get("image_occurrence_matched.episode")
        == verified["episode_source_schedule"]["sha256"]
        and final_schedules.get("image_occurrence_matched.isolated")
        == verified["isolated_source_schedule"]["sha256"],
        "final release index does not bind the pilot parent schedules",
    )
    compute_sources = compute_manifest.get("sources")
    _expect(isinstance(compute_sources, dict), "compute manifest has no source bindings")
    compute_episode_source = compute_sources.get("episode")
    _expect(
        isinstance(compute_episode_source, dict),
        "compute manifest has no episode source binding",
    )
    declared_episode_source_path = (
        Path(verified["compute_matching_manifest"]["path"]).parent
        / str(compute_episode_source.get("path", ""))
    ).resolve()
    _expect(
        declared_episode_source_path
        == Path(verified["episode_selection_sft"]["path"])
        and compute_episode_source.get("sha256")
        == verified["episode_selection_sft"]["sha256"],
        "compute manifest does not bind the episode selection SFT",
    )
    regime = compute_manifest.get("regimes", {}).get("image_occurrence_matched", {})
    files = regime.get("schedule_files", {})
    for arm, lineage_name in (
        ("episode", "episode_source_schedule"),
        ("isolated", "isolated_source_schedule"),
    ):
        declared = files.get(arm, {})
        declared_path = (
            Path(verified["compute_matching_manifest"]["path"]).parent
            / str(declared.get("filename", ""))
        ).resolve()
        _expect(
            declared_path == Path(verified[lineage_name]["path"]),
            f"compute manifest {arm} schedule path mismatch",
        )
        _expect(
            declared.get("sha256") == verified[lineage_name]["sha256"],
            f"compute manifest {arm} schedule digest mismatch",
        )
    return {"artifacts": verified, "checks": dict(sorted(raw_checks.items()))}


def _validate_schedule_binding(
    layout: GroupStepPilotLayout,
) -> tuple[
    list[ScheduledRecord],
    list[ScheduledRecord],
    dict[str, ComparisonGroup],
    dict[str, ComparisonGroup],
    dict[str, int],
    dict[str, Any],
]:
    _expect(layout.episode_schedule.is_file(), "episode schedule is missing")
    _expect(layout.isolated_schedule.is_file(), "isolated schedule is missing")
    episode_items = load_scheduled_records(layout.episode_schedule)
    isolated_items = load_scheduled_records(layout.isolated_schedule)
    episode_groups = build_comparison_groups(episode_items)
    isolated_groups = build_comparison_groups(isolated_items)
    paired_validation = validate_paired_comparison_groups(
        episode_groups, isolated_groups
    )
    episode_sha = sha256(layout.episode_schedule)
    isolated_sha = sha256(layout.isolated_schedule)

    schedule_manifest = _load_json(layout.schedule_manifest)
    _expect(schedule_manifest.get("status") == "pass", "schedule manifest did not pass")
    bindings = schedule_manifest.get("artifacts")
    _expect(isinstance(bindings, dict), "schedule manifest has no artifacts")
    for arm, path, observed_sha in (
        ("episode", layout.episode_schedule, episode_sha),
        ("isolated", layout.isolated_schedule, isolated_sha),
    ):
        binding = bindings.get(arm)
        _expect(isinstance(binding, dict), f"schedule manifest has no {arm} binding")
        bound_path = (layout.schedule_manifest.parent / str(binding.get("path", ""))).resolve()
        _expect(bound_path == path.resolve(), f"schedule manifest {arm} path mismatch")
        _expect(binding.get("sha256") == observed_sha, f"schedule manifest {arm} is stale")

    token_profile = _load_json(layout.token_profile)
    _expect(token_profile.get("status") == "pass", "token profile did not pass")
    _expect(
        Path(str(token_profile.get("model", ""))).resolve() == layout.model.resolve(),
        "token profile model mismatch",
    )
    _expect(
        token_profile.get("image_min_pixels")
        == layout.expected_training_protocol["image_min_pixels"],
        "token profile image_min_pixels differs from the frozen training protocol",
    )
    _expect(
        token_profile.get("image_max_pixels")
        == layout.expected_training_protocol["image_max_pixels"],
        "token profile image_max_pixels differs from the frozen training protocol",
    )
    profile_sources = token_profile.get("sources")
    _expect(isinstance(profile_sources, dict), "token profile has no source hashes")
    _expect(
        profile_sources.get("episode_schedule_sha256") == episode_sha,
        "token profile episode schedule is stale",
    )
    _expect(
        profile_sources.get("isolated_schedule_sha256") == isolated_sha,
        "token profile isolated schedule is stale",
    )
    checks = token_profile.get("checks")
    _expect(isinstance(checks, dict), "token profile has no matching checks")
    for key in ("image_tokens_equal", "effective_facts_equal", "effective_fact_weights_one"):
        _expect(checks.get(key) is True, f"token profile failed {key}")
    profile_episode = token_profile.get("episode")
    profile_isolated = token_profile.get("isolated")
    _expect(
        isinstance(profile_episode, dict) and isinstance(profile_isolated, dict),
        "token profile has no arm totals",
    )

    episode_surface = _surface_contract(episode_items, "episode")
    isolated_surface = _surface_contract(isolated_items, "isolated")
    _expect(episode_surface == isolated_surface, "training arm surface formats differ")
    _expect(
        episode_surface == layout.expected_presentation_contract,
        "training surface differs from the declared presentation contract",
    )
    release_lineage = _validate_pilot_release_lineage(
        layout.schedule_manifest, schedule_manifest
    )
    matching = {
        "comparison_groups": paired_validation["comparison_groups"],
        "draws_per_arm": paired_validation["draws_per_arm"],
        "facts": paired_validation["facts"],
        "surface_format": episode_surface,
        "episode_schedule_sha256": episode_sha,
        "isolated_schedule_sha256": isolated_sha,
        "release_lineage": release_lineage,
        "token_profile": {
            "image_min_pixels": token_profile["image_min_pixels"],
            "image_max_pixels": token_profile["image_max_pixels"],
            "episode": profile_episode,
            "isolated": profile_isolated,
            "checks": checks,
        },
    }
    return (
        episode_items,
        isolated_items,
        episode_groups,
        isolated_groups,
        paired_validation,
        matching,
    )


def _validate_training_run(
    *,
    train_dir: Path,
    arm: str,
    seed: int,
    own_schedule: Path,
    paired_schedule: Path,
    own_groups: dict[str, ComparisonGroup],
    paired_validation: dict[str, int],
    model: Path,
    expected_model_inventory_sha256: str,
) -> dict[str, Any]:
    manifest_path = train_dir / "run_manifest.json"
    log_path = train_dir / "train_log.jsonl"
    adapter_dir = train_dir / "adapter"
    manifest = _load_json(manifest_path)
    _expect(
        manifest.get("schema_version") == TRAIN_MANIFEST_SCHEMA_VERSION,
        f"seed {seed} {arm} is not a v2 group-step run",
    )
    _expect(manifest.get("status") == "complete", f"seed {seed} {arm} is incomplete")
    _expect(
        manifest.get("optimizer_unit") == "comparison_group",
        f"seed {seed} {arm} did not use comparison-group updates",
    )
    _expect(manifest.get("world_size") == 1, f"seed {seed} {arm} world_size is not one")
    config = manifest.get("config")
    _expect(isinstance(config, dict), f"seed {seed} {arm} has no training config")
    _expect(
        config.get("optimizer_unit") == "comparison_group",
        f"seed {seed} {arm} config is not comparison_group",
    )
    _expect(
        config.get("gradient_accumulation_steps") == 1,
        f"seed {seed} {arm} gradient accumulation changed the optimizer unit",
    )
    _expect(config.get("max_steps") is None, f"seed {seed} {arm} is a truncated run")
    _expect(config.get("seed") == seed, f"seed {seed} {arm} seed mismatch")
    _expect(
        Path(str(config.get("model", ""))).resolve() == model.resolve(),
        f"seed {seed} {arm} base model mismatch",
    )
    base_model_provenance = validate_inventory_binding(
        manifest.get("base_model_provenance"),
        root=model,
        kind="model",
        expected_inventory_sha256=expected_model_inventory_sha256,
    )
    _expect(
        manifest.get("base_model_inventory_sha256")
        == expected_model_inventory_sha256,
        f"seed {seed} {arm} base model inventory digest mismatch",
    )
    _expect(
        Path(str(config.get("model_inventory", ""))).resolve()
        == Path(base_model_provenance["inventory_artifact"]).resolve(),
        f"seed {seed} {arm} model inventory artifact mismatch",
    )
    _expect(
        Path(str(config.get("schedule", ""))).resolve() == own_schedule.resolve(),
        f"seed {seed} {arm} schedule path mismatch",
    )
    declared_pair = config.get("paired_schedule")
    if declared_pair is not None:
        _expect(
            Path(str(declared_pair)).resolve() == paired_schedule.resolve(),
            f"seed {seed} {arm} paired schedule path mismatch",
        )
    declared_output = config.get("output_dir")
    _expect(
        isinstance(declared_output, str)
        and Path(declared_output).resolve() == train_dir.resolve(),
        f"seed {seed} {arm} output directory mismatch",
    )
    _expect(
        manifest.get("schedule_sha256") == sha256(own_schedule),
        f"seed {seed} {arm} own schedule hash mismatch",
    )
    _expect(
        manifest.get("paired_schedule_sha256") == sha256(paired_schedule),
        f"seed {seed} {arm} paired schedule hash mismatch",
    )
    epochs = config.get("epochs")
    _expect(
        isinstance(epochs, int) and not isinstance(epochs, bool) and epochs >= 1,
        f"seed {seed} {arm} epochs are invalid",
    )
    draws = sum(group.draw_count for group in own_groups.values())
    group_count = len(own_groups)
    expected_updates = group_count * epochs
    expected_micro_steps = draws * epochs
    expected_completed_groups = expected_updates
    _expect(manifest.get("draws") == draws, f"seed {seed} {arm} draw count mismatch")
    _expect(
        manifest.get("comparison_groups") == group_count,
        f"seed {seed} {arm} comparison group count mismatch",
    )
    _expect(
        manifest.get("groups_per_epoch") == group_count,
        f"seed {seed} {arm} groups_per_epoch mismatch",
    )
    _expect(
        manifest.get("paired_group_validation") == paired_validation,
        f"seed {seed} {arm} paired group validation mismatch",
    )
    _expect(
        manifest.get("local_draws_per_epoch") == draws,
        f"seed {seed} {arm} local draw count mismatch",
    )
    _expect(
        manifest.get("planned_optimizer_updates") == expected_updates,
        f"seed {seed} {arm} planned optimizer updates mismatch",
    )
    _expect(
        manifest.get("optimizer_updates") == expected_updates,
        f"seed {seed} {arm} optimizer updates are not groups x epochs",
    )
    _expect(
        manifest.get("completed_comparison_groups") == expected_completed_groups,
        f"seed {seed} {arm} did not complete every comparison group",
    )
    _expect(
        manifest.get("completed_micro_steps_per_rank") == expected_micro_steps,
        f"seed {seed} {arm} microsteps are not draws x epochs",
    )
    loss_contract = manifest.get("loss_contract")
    _expect(
        isinstance(loss_contract, str)
        and "before exactly one optimizer step" in loss_contract,
        f"seed {seed} {arm} does not declare the group-step loss contract",
    )
    _expect(log_path.is_file(), f"seed {seed} {arm} training log is missing")
    _expect(adapter_dir.is_dir(), f"seed {seed} {arm} adapter is missing")
    adapter_inventory = directory_inventory(adapter_dir)
    adapter_provenance = validate_inventory_binding(
        manifest.get("adapter_provenance"),
        root=adapter_dir,
        kind="directory",
        expected_inventory_sha256=adapter_inventory["inventory_sha256"],
    )
    _expect(
        manifest.get("adapter_inventory_sha256")
        == adapter_inventory["inventory_sha256"],
        f"seed {seed} {arm} adapter inventory digest mismatch",
    )
    protocol = {key: config.get(key) for key in TRAIN_PROTOCOL_KEYS}
    _expect(
        all(value is not None for value in protocol.values()),
        f"seed {seed} {arm} training protocol is incomplete",
    )
    return {
        "manifest": manifest,
        "protocol": protocol,
        "epochs": epochs,
        "weight_provenance": {
            "base_model": base_model_provenance,
            "adapter": adapter_provenance,
        },
        "artifacts": {
            "training_manifest": _artifact(manifest_path),
            "training_log": _artifact(log_path),
            "adapter": adapter_inventory,
        },
    }


def _metric(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "status": "defined" if total else "NA_ZERO_DENOMINATOR",
    }


def _group_metric(
    rows: list[dict[str, Any]],
    correctness: dict[str, bool],
    key,
) -> dict[str, dict[str, Any]]:
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        label = str(key(row))
        totals[label] += 1
        correct[label] += int(correctness[str(row["record_id"])])
    return {
        label: _metric(correct[label], totals[label]) for label in sorted(totals)
    }


def _split_metrics(
    *,
    benchmark_rows: list[dict[str, Any]],
    correctness: dict[str, bool],
    parsed_by_id: dict[str, bool],
    split: str,
    composition_program_ids: frozenset[str],
) -> dict[str, Any]:
    """Recompute all split-sensitive endpoints from record-level correctness."""

    rows = [row for row in benchmark_rows if row.get("split") == split]
    _expect(rows, f"benchmark has no {split} records")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row.get("consistency_group")
        if isinstance(group, str) and group:
            groups[group].append(row)

    eligible: dict[str, list[list[dict[str, Any]]]] = {
        name: [] for name in SIBLING_VARIANTS
    }
    for group_rows in groups.values():
        variants = frozenset(str(row.get("family_variant")) for row in group_rows)
        for name, expected in SIBLING_VARIANTS.items():
            if variants == expected and len(group_rows) == len(expected):
                eligible[name].append(group_rows)
                break
    all_families = [group for groups_for_type in eligible.values() for group in groups_for_type]

    def family_exact(group_rows: list[dict[str, Any]]) -> bool:
        return all(correctness[str(row["record_id"])] for row in group_rows)

    family_metric = _metric(
        sum(family_exact(group) for group in all_families), len(all_families)
    )
    claim_metric = _metric(
        sum(family_exact(group) for group in eligible["claim_pair"]),
        len(eligible["claim_pair"]),
    )
    frame_metric = _metric(
        sum(family_exact(group) for group in eligible["frame_pair"]),
        len(eligible["frame_pair"]),
    )
    revealed_correct = 0
    evidence_exact = 0
    evidence_details: list[dict[str, Any]] = []
    for group_rows in eligible["evidence_triple"]:
        by_variant = {str(row["family_variant"]): row for row in group_rows}
        revealed = correctness[str(by_variant["revealed"]["record_id"])]
        joint = family_exact(group_rows)
        revealed_correct += int(revealed)
        evidence_exact += int(joint)
        evidence_details.append(
            {
                "consistency_group": str(group_rows[0]["consistency_group"]),
                "revealed_correct": revealed,
                "exact_match": joint,
            }
        )
    evidence_total = len(eligible["evidence_triple"])
    composition_rows = [
        row
        for row in rows
        if row.get("program", {}).get("program_id")
        in composition_program_ids
    ]
    composition_program_counts = Counter(
        str(row["program"]["program_id"]) for row in composition_rows
    )
    return {
        "records": len(rows),
        "record_accuracy": _metric(
            sum(correctness[str(row["record_id"])] for row in rows), len(rows)
        ),
        "parse_rate": _metric(
            sum(parsed_by_id[str(row["record_id"])] for row in rows), len(rows)
        ),
        "heldout_composition": {
            **_metric(
                sum(correctness[str(row["record_id"])] for row in composition_rows),
                len(composition_rows),
            ),
            "program_counts": dict(sorted(composition_program_counts.items())),
        },
        "per_program": _group_metric(
            rows, correctness, lambda row: row["program"]["program_id"]
        ),
        "family_exact_match": family_metric,
        "claim_pair_exact_match": claim_metric,
        "frame_pair_exact_match": frame_metric,
        "revealed_accuracy": _metric(revealed_correct, evidence_total),
        "evidence_triple_exact_match": _metric(evidence_exact, evidence_total),
        "aces": {
            "numerator_joint_correct": evidence_exact,
            "denominator_revealed_correct": revealed_correct,
            "value": evidence_exact / revealed_correct if revealed_correct else None,
            "status": "defined" if revealed_correct else "NA_ZERO_DENOMINATOR",
        },
        "group_counts": {
            "families": len(all_families),
            "claim_pairs": len(eligible["claim_pair"]),
            "frame_pairs": len(eligible["frame_pair"]),
            "evidence_triples": evidence_total,
        },
        "evidence_details": evidence_details,
    }


def _validate_split_reconstruction(
    evaluation: dict[str, Any], by_split: dict[str, dict[str, Any]]
) -> None:
    """Prove that reporter-side strata exactly reconstruct evaluator v2 totals."""

    for split, metrics in by_split.items():
        _expect(
            evaluation.get("per_split", {}).get(split) == metrics["record_accuracy"],
            f"reporter/evaluator {split} record metric mismatch",
        )
    aggregate_pairs = (
        ("family_exact_match", evaluation["family_exact_match"]),
        ("claim_pair_exact_match", evaluation["claim_pair_exact_match"]),
        (
            "frame_pair_exact_match",
            evaluation["frame_equivariance"]["frame_pair_exact_match"],
        ),
        ("revealed_accuracy", evaluation["evidence_triples"]["revealed_accuracy"]),
        (
            "evidence_triple_exact_match",
            evaluation["evidence_triples"]["exact_match"],
        ),
    )
    for metric_name, evaluator_metric in aggregate_pairs:
        correct = sum(split[metric_name]["correct"] for split in by_split.values())
        total = sum(split[metric_name]["total"] for split in by_split.values())
        _expect(
            correct == evaluator_metric["correct"] and total == evaluator_metric["total"],
            f"reporter/evaluator aggregate mismatch for {metric_name}",
        )
    joint = sum(split["aces"]["numerator_joint_correct"] for split in by_split.values())
    revealed = sum(
        split["aces"]["denominator_revealed_correct"] for split in by_split.values()
    )
    evaluator_aces = evaluation["evidence_triples"][
        "accuracy_conditioned_evidence_sensitivity"
    ]
    _expect(
        joint == evaluator_aces["numerator_joint_correct"]
        and revealed == evaluator_aces["denominator_revealed_correct"],
        "reporter/evaluator aggregate mismatch for ACES",
    )


def _collect_inference(
    *,
    name: str,
    inference_dir: Path,
    benchmark: Path,
    benchmark_sha: str,
    benchmark_rows: list[dict[str, Any]],
    benchmark_ids: set[str],
    composition_program_counts: dict[str, int],
    composition_record_ids_by_program: dict[str, list[str]],
    model: Path,
    expected_model_inventory_sha256: str,
    expected_prompt_mode: str,
    expected_image_mode: str,
    adapter_dir: Path | None,
    expected_adapter_inventory_sha256: str | None,
) -> dict[str, Any]:
    manifest_path = inference_dir / "prediction_manifest.json"
    predictions_path = inference_dir / "predictions.jsonl"
    raw_path = inference_dir / "raw_generations.jsonl"
    evaluation_path = inference_dir / "evaluation" / "evaluation.json"
    evaluation_md_path = inference_dir / "evaluation" / "evaluation.md"
    manifest = _load_json(manifest_path)
    predictions = _read_jsonl(predictions_path)
    raw = _read_jsonl(raw_path)
    evaluation = _load_json(evaluation_path)

    _expect(
        manifest.get("schema_version") == PREDICTION_MANIFEST_SCHEMA_VERSION,
        f"{name} has unsupported prediction manifest schema",
    )
    _expect(manifest.get("num_shards") == 1, f"{name} inference is not merged")
    _expect(
        Path(str(manifest.get("model", ""))).resolve() == model.resolve(),
        f"{name} inference model mismatch",
    )
    base_model_provenance = validate_inventory_binding(
        manifest.get("base_model_provenance"),
        root=model,
        kind="model",
        expected_inventory_sha256=expected_model_inventory_sha256,
    )
    _expect(
        manifest.get("base_model_inventory_sha256")
        == expected_model_inventory_sha256,
        f"{name} base model inventory digest mismatch",
    )
    _expect(
        Path(str(manifest.get("benchmark", ""))).resolve() == benchmark.resolve(),
        f"{name} inference benchmark path mismatch",
    )
    _expect(
        manifest.get("benchmark_sha256") == benchmark_sha,
        f"{name} inference benchmark is stale",
    )
    _expect(
        manifest.get("prompt_mode") == expected_prompt_mode,
        f"{name} prompt mode differs from the formal contract",
    )
    _expect(manifest.get("image_mode") == expected_image_mode, f"{name} image mode mismatch")
    expected_adapter = adapter_dir.resolve() if adapter_dir is not None else None
    declared_adapter = manifest.get("adapter")
    observed_adapter = Path(str(declared_adapter)).resolve() if declared_adapter else None
    _expect(observed_adapter == expected_adapter, f"{name} adapter binding mismatch")
    if adapter_dir is None:
        _expect(
            manifest.get("adapter_inventory_sha256") is None
            and manifest.get("adapter_provenance") is None
            and expected_adapter_inventory_sha256 is None,
            f"{name} adapter-free inference declares adapter provenance",
        )
        adapter_provenance = None
    else:
        _expect(
            isinstance(expected_adapter_inventory_sha256, str),
            f"{name} has no expected adapter inventory digest",
        )
        adapter_provenance = validate_inventory_binding(
            manifest.get("adapter_provenance"),
            root=adapter_dir,
            kind="directory",
            expected_inventory_sha256=expected_adapter_inventory_sha256,
        )
        _expect(
            manifest.get("adapter_inventory_sha256")
            == expected_adapter_inventory_sha256,
            f"{name} adapter inventory digest mismatch",
        )
    _expect(
        Path(str(manifest.get("predictions", ""))).resolve() == predictions_path.resolve(),
        f"{name} predictions path mismatch",
    )
    _expect(
        Path(str(manifest.get("raw_generations", ""))).resolve() == raw_path.resolve(),
        f"{name} raw generations path mismatch",
    )
    predictions_sha = sha256(predictions_path)
    raw_sha = sha256(raw_path)
    _expect(
        manifest.get("predictions_sha256") == predictions_sha,
        f"{name} predictions hash mismatch",
    )
    _expect(
        manifest.get("raw_generations_sha256") == raw_sha,
        f"{name} raw generations hash mismatch",
    )
    record_count = len(benchmark_rows)
    _expect(manifest.get("records") == record_count, f"{name} record count mismatch")
    _expect(len(predictions) == record_count, f"{name} predictions count mismatch")
    _expect(len(raw) == record_count, f"{name} raw generation count mismatch")
    prediction_ids = [row.get("record_id") for row in predictions]
    raw_ids = [row.get("record_id") for row in raw]
    _expect(len(set(prediction_ids)) == record_count, f"{name} has duplicate prediction IDs")
    _expect(set(prediction_ids) == benchmark_ids, f"{name} predictions do not cover benchmark")
    _expect(set(raw_ids) == benchmark_ids, f"{name} raw generations do not cover benchmark")
    parsed = sum(row.get("parse_status") == "parsed" for row in predictions)
    _expect(manifest.get("parsed") == parsed, f"{name} parsed count mismatch")
    parsed_by_id = {
        str(row["record_id"]): row.get("parse_status") == "parsed" for row in predictions
    }

    _expect(
        evaluation.get("schema_version") == EVALUATION_SCHEMA_VERSION,
        f"{name} has unsupported evaluator schema",
    )
    inputs = evaluation.get("inputs")
    _expect(isinstance(inputs, dict), f"{name} evaluation has no input binding")
    _expect(inputs.get("benchmark_sha256") == benchmark_sha, f"{name} evaluator is stale")
    _expect(inputs.get("predictions_sha256") == predictions_sha, f"{name} evaluator is stale")
    _expect(
        Path(str(inputs.get("benchmark", ""))).resolve() == benchmark.resolve(),
        f"{name} evaluator benchmark path mismatch",
    )
    _expect(
        Path(str(inputs.get("predictions", ""))).resolve() == predictions_path.resolve(),
        f"{name} evaluator predictions path mismatch",
    )
    coverage = evaluation.get("predictions")
    _expect(isinstance(coverage, dict), f"{name} evaluation has no coverage")
    for key in ("missing_count", "extra_count", "duplicate_id_count"):
        _expect(coverage.get(key) == 0, f"{name} evaluator reports nonzero {key}")
    _expect(
        evaluation.get("benchmark", {}).get("records") == record_count,
        f"{name} evaluator record count mismatch",
    )
    record_results = evaluation.get("record_results")
    _expect(isinstance(record_results, list), f"{name} evaluator has no record results")
    correctness: dict[str, bool] = {}
    for result in record_results:
        _expect(isinstance(result, dict), f"{name} has malformed record result")
        record_id = result.get("record_id")
        correct = result.get("correct")
        _expect(
            isinstance(record_id, str) and record_id in benchmark_ids,
            f"{name} evaluator contains unknown record",
        )
        _expect(record_id not in correctness, f"{name} evaluator has duplicate record result")
        _expect(isinstance(correct, bool), f"{name} correctness is not boolean")
        correctness[record_id] = correct
    _expect(set(correctness) == benchmark_ids, f"{name} evaluator results are incomplete")
    _expect(
        sum(correctness.values()) == evaluation.get("record_accuracy", {}).get("correct"),
        f"{name} record results disagree with record accuracy",
    )
    metrics = extract_metrics(evaluation, composition_program_counts)
    for program_id, record_ids in composition_record_ids_by_program.items():
        _expect(
            metrics["per_program"][program_id]["correct"]
            == sum(correctness[record_id] for record_id in record_ids),
            f"{name} composition metric disagrees with record results",
        )
    by_split = {
        split: _split_metrics(
            benchmark_rows=benchmark_rows,
            correctness=correctness,
            parsed_by_id=parsed_by_id,
            split=split,
            composition_program_ids=frozenset(composition_program_counts),
        )
        for split in ("val", "test")
    }
    _validate_split_reconstruction(evaluation, by_split)
    inference_protocol = {key: manifest.get(key) for key in INFERENCE_PROTOCOL_KEYS}
    _expect(
        all(value is not None for value in inference_protocol.values()),
        f"{name} inference protocol is incomplete",
    )
    return {
        "status": "complete",
        "image_mode": expected_image_mode,
        "parsed": parsed,
        "records": record_count,
        "parse_rate": parsed / record_count,
        "inference_protocol": inference_protocol,
        "weight_provenance": {
            "base_model": base_model_provenance,
            "adapter": adapter_provenance,
        },
        "metrics": metrics,
        "metrics_by_split": by_split,
        "record_correctness": correctness,
        "artifacts": {
            "prediction_manifest": _artifact(manifest_path),
            "predictions": _artifact(predictions_path),
            "raw_generations": _artifact(raw_path),
            "evaluation_json": _artifact(evaluation_path),
            "evaluation_markdown": _artifact(evaluation_md_path),
        },
    }


def _scalar_metrics(run: dict[str, Any]) -> dict[str, float | None]:
    metrics = run["metrics"]
    values: dict[str, float | None] = {
        "record_accuracy": metrics["record_accuracy"]["accuracy"],
        "family_exact_match": metrics["family_exact_match"]["accuracy"],
        "claim_pair_exact_match": metrics["claim_pair_exact_match"]["accuracy"],
        "frame_pair_exact_match": metrics["frame_pair_exact_match"]["accuracy"],
        "revealed_accuracy": metrics["revealed_accuracy"]["accuracy"],
        "evidence_triple_exact_match": metrics["evidence_triple_exact_match"]["accuracy"],
        "aces": metrics["aces"]["value"],
    }
    for split in ("test", "val"):
        stratum = run["metrics_by_split"][split]
        values.update(
            {
                f"{split}_record_accuracy": stratum["record_accuracy"]["accuracy"],
                f"{split}_heldout_composition": stratum["heldout_composition"][
                    "accuracy"
                ],
                f"{split}_family_exact_match": stratum["family_exact_match"][
                    "accuracy"
                ],
                f"{split}_claim_pair_exact_match": stratum["claim_pair_exact_match"][
                    "accuracy"
                ],
                f"{split}_frame_pair_exact_match": stratum["frame_pair_exact_match"][
                    "accuracy"
                ],
                f"{split}_revealed_accuracy": stratum["revealed_accuracy"][
                    "accuracy"
                ],
                f"{split}_evidence_triple_exact_match": stratum[
                    "evidence_triple_exact_match"
                ]["accuracy"],
                f"{split}_aces": stratum["aces"]["value"],
            }
        )
    return values


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_values = _scalar_metrics(left)
    right_values = _scalar_metrics(right)
    aggregate = {
        key: left_values[key] - right_values[key]
        if left_values[key] is not None and right_values[key] is not None
        else None
        for key in AGGREGATE_METRICS
    }
    left_metrics = left["metrics"]
    right_metrics = right["metrics"]
    programs = sorted(
        set(left_metrics["per_program"]) | set(right_metrics["per_program"])
    )
    per_program: dict[str, float | None] = {}
    for program in programs:
        left_value = left_metrics["per_program"].get(program, {}).get("accuracy")
        right_value = right_metrics["per_program"].get(program, {}).get("accuracy")
        per_program[program] = (
            left_value - right_value
            if left_value is not None and right_value is not None
            else None
        )
    return {"aggregate": aggregate, "per_program": per_program}


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _scene_bootstrap(
    *,
    benchmark_rows: list[dict[str, Any]],
    record_ids: set[str],
    episode_correctness: dict[str, bool],
    isolated_correctness: dict[str, bool],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    clusters: dict[str, list[str]] = {}
    for row in benchmark_rows:
        record_id = row.get("record_id")
        if record_id not in record_ids:
            continue
        scene_id = row.get("scene_id")
        _expect(isinstance(scene_id, str) and scene_id, f"record {record_id} has no scene_id")
        clusters.setdefault(scene_id, []).append(str(record_id))
    _expect(
        {record_id for ids in clusters.values() for record_id in ids} == record_ids,
        "scene bootstrap does not cover its endpoint",
    )
    scene_ids = sorted(clusters)
    _expect(len(scene_ids) >= 2, "scene bootstrap requires at least two scenes")
    differences = {
        record_id: int(episode_correctness[record_id])
        - int(isolated_correctness[record_id])
        for record_id in record_ids
    }
    observed = sum(differences.values()) / len(record_ids)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        numerator = 0
        denominator = 0
        for _ in scene_ids:
            sampled_scene = scene_ids[rng.randrange(len(scene_ids))]
            sampled_records = clusters[sampled_scene]
            numerator += sum(differences[record_id] for record_id in sampled_records)
            denominator += len(sampled_records)
        samples.append(numerator / denominator)
    samples.sort()
    return {
        "method": "paired nonparametric percentile bootstrap over scene clusters",
        "observed_delta": observed,
        "ci95_percentile": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "scene_clusters": len(scene_ids),
        "records": len(record_ids),
        "seed": seed,
        "resamples": resamples,
        "interpretation": "benchmark-scene sampling uncertainty within one training seed",
    }


def _summarize_values(values_by_seed: dict[str, float | None]) -> dict[str, Any]:
    defined = [float(value) for value in values_by_seed.values() if value is not None]
    return {
        "values_by_seed": values_by_seed,
        "defined_seeds": len(defined),
        "mean": sum(defined) / len(defined) if defined else None,
        "min": min(defined) if defined else None,
        "max": max(defined) if defined else None,
    }


def _cross_seed_summary(seed_runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"episode": {}, "isolated": {}, "episode_minus_isolated": {}}
    for metric in AGGREGATE_METRICS:
        episode_values = {
            seed: _scalar_metrics(run["episode"])[metric]
            for seed, run in seed_runs.items()
        }
        isolated_values = {
            seed: _scalar_metrics(run["isolated"])[metric]
            for seed, run in seed_runs.items()
        }
        delta_values = {
            seed: run["episode_minus_isolated"]["aggregate"][metric]
            for seed, run in seed_runs.items()
        }
        result["episode"][metric] = _summarize_values(episode_values)
        result["isolated"][metric] = _summarize_values(isolated_values)
        result["episode_minus_isolated"][metric] = _summarize_values(delta_values)
    programs = sorted(
        {
            program
            for run in seed_runs.values()
            for program in run["episode"]["metrics"]["per_program"]
        }
    )
    result["per_program"] = {}
    for program in programs:
        result["per_program"][program] = {
            arm: _summarize_values(
                {
                    seed: (
                        run[arm]["metrics"]["per_program"].get(program, {}).get("accuracy")
                        if arm in {"episode", "isolated"}
                        else run["episode_minus_isolated"]["per_program"].get(program)
                    )
                    for seed, run in seed_runs.items()
                }
            )
            for arm in ("episode", "isolated", "episode_minus_isolated")
        }
    result["per_split_program"] = {}
    for split in ("test", "val"):
        split_programs = sorted(
            {
                program
                for run in seed_runs.values()
                for program in run["episode"]["metrics_by_split"][split]["per_program"]
            }
        )
        result["per_split_program"][split] = {}
        for program in split_programs:
            episode_values = {
                seed: run["episode"]["metrics_by_split"][split]["per_program"]
                .get(program, {})
                .get("accuracy")
                for seed, run in seed_runs.items()
            }
            isolated_values = {
                seed: run["isolated"]["metrics_by_split"][split]["per_program"]
                .get(program, {})
                .get("accuracy")
                for seed, run in seed_runs.items()
            }
            delta_values = {
                seed: (
                    episode_values[seed] - isolated_values[seed]
                    if episode_values[seed] is not None
                    and isolated_values[seed] is not None
                    else None
                )
                for seed in seed_runs
            }
            result["per_split_program"][split][program] = {
                "episode": _summarize_values(episode_values),
                "isolated": _summarize_values(isolated_values),
                "episode_minus_isolated": _summarize_values(delta_values),
            }
    return result


def _formal_summary(layout: GroupStepPilotLayout) -> dict[str, Any]:
    benchmark_rows = _read_jsonl(layout.benchmark)
    composition_rows = _read_jsonl(layout.composition_benchmark)
    _expect(benchmark_rows, "benchmark is empty")
    benchmark_ids = [row.get("record_id") for row in benchmark_rows]
    _expect(
        all(isinstance(record_id, str) and record_id for record_id in benchmark_ids),
        "benchmark contains an invalid record ID",
    )
    _expect(len(set(benchmark_ids)) == len(benchmark_ids), "benchmark IDs are not unique")
    consistency_splits: dict[str, set[str]] = defaultdict(set)
    for row in benchmark_rows:
        group = row.get("consistency_group")
        if isinstance(group, str) and group:
            consistency_splits[group].add(str(row.get("split")))
    _expect(
        all(len(splits) == 1 for splits in consistency_splits.values()),
        "a sibling consistency group crosses val/test splits",
    )
    surfaces = {row.get("presentation_contract") for row in benchmark_rows}
    _expect(
        surfaces == {layout.expected_presentation_contract},
        f"benchmark presentation contracts differ from {layout.expected_presentation_contract}",
    )
    composition_contract = validate_composition_benchmark(
        benchmark_rows, composition_rows
    )
    benchmark_sha = sha256(layout.benchmark)
    current_model_inventory = model_inventory(layout.model)
    model_inventory_sha256 = current_model_inventory["inventory_sha256"]
    _expect(
        benchmark_sha == layout.expected_benchmark_sha256,
        "benchmark hash differs from the frozen report config",
    )
    primary_rows = [
        row
        for row in benchmark_rows
        if row.get("split") == PRIMARY_COMPOSITION_SPLIT
        and row.get("program", {}).get("program_id")
        in layout.expected_primary_composition_program_counts
    ]
    primary_program_counts = Counter(
        str(row["program"]["program_id"]) for row in primary_rows
    )
    _expect(
        dict(primary_program_counts)
        == layout.expected_primary_composition_program_counts,
        "observed test-only composition counts differ from the frozen config",
    )
    primary_record_ids = {str(row["record_id"]) for row in primary_rows}
    (
        _episode_items,
        _isolated_items,
        episode_groups,
        isolated_groups,
        paired_validation,
        matching,
    ) = _validate_schedule_binding(layout)

    seed_runs: dict[str, dict[str, Any]] = {}
    shared_training_protocol: dict[str, Any] | None = None
    shared_inference_protocol: dict[str, Any] | None = None
    all_record_ids = set(str(record_id) for record_id in benchmark_ids)
    for seed_layout in layout.seeds:
        episode_training = _validate_training_run(
            train_dir=seed_layout.episode.train_dir,
            arm="episode",
            seed=seed_layout.seed,
            own_schedule=layout.episode_schedule,
            paired_schedule=layout.isolated_schedule,
            own_groups=episode_groups,
            paired_validation=paired_validation,
            model=layout.model,
            expected_model_inventory_sha256=model_inventory_sha256,
        )
        isolated_training = _validate_training_run(
            train_dir=seed_layout.isolated.train_dir,
            arm="isolated",
            seed=seed_layout.seed,
            own_schedule=layout.isolated_schedule,
            paired_schedule=layout.episode_schedule,
            own_groups=isolated_groups,
            paired_validation=paired_validation,
            model=layout.model,
            expected_model_inventory_sha256=model_inventory_sha256,
        )
        _expect(
            episode_training["protocol"] == isolated_training["protocol"],
            f"seed {seed_layout.seed} training hyperparameters differ across arms",
        )
        if shared_training_protocol is None:
            shared_training_protocol = episode_training["protocol"]
        else:
            _expect(
                episode_training["protocol"] == shared_training_protocol,
                f"seed {seed_layout.seed} changes the shared training protocol",
            )
        expected_protocol = {
            "model": str(layout.model.resolve()),
            **layout.expected_training_protocol,
        }
        _expect(
            episode_training["protocol"] == expected_protocol,
            f"seed {seed_layout.seed} differs from the frozen training protocol",
        )

        episode_inference = _collect_inference(
            name=f"seed {seed_layout.seed} episode",
            inference_dir=seed_layout.episode.inference_dir,
            benchmark=layout.benchmark,
            benchmark_sha=benchmark_sha,
            benchmark_rows=benchmark_rows,
            benchmark_ids=all_record_ids,
            composition_program_counts=composition_contract["program_counts"],
            composition_record_ids_by_program=composition_contract[
                "record_ids_by_program"
            ],
            model=layout.model,
            expected_model_inventory_sha256=model_inventory_sha256,
            expected_prompt_mode=layout.expected_prompt_mode,
            expected_image_mode="full",
            adapter_dir=seed_layout.episode.train_dir / "adapter",
            expected_adapter_inventory_sha256=episode_training["artifacts"][
                "adapter"
            ]["inventory_sha256"],
        )
        isolated_inference = _collect_inference(
            name=f"seed {seed_layout.seed} isolated",
            inference_dir=seed_layout.isolated.inference_dir,
            benchmark=layout.benchmark,
            benchmark_sha=benchmark_sha,
            benchmark_rows=benchmark_rows,
            benchmark_ids=all_record_ids,
            composition_program_counts=composition_contract["program_counts"],
            composition_record_ids_by_program=composition_contract[
                "record_ids_by_program"
            ],
            model=layout.model,
            expected_model_inventory_sha256=model_inventory_sha256,
            expected_prompt_mode=layout.expected_prompt_mode,
            expected_image_mode="full",
            adapter_dir=seed_layout.isolated.train_dir / "adapter",
            expected_adapter_inventory_sha256=isolated_training["artifacts"][
                "adapter"
            ]["inventory_sha256"],
        )
        _expect(
            episode_inference["inference_protocol"]
            == isolated_inference["inference_protocol"],
            f"seed {seed_layout.seed} inference protocols differ across arms",
        )
        if shared_inference_protocol is None:
            shared_inference_protocol = episode_inference["inference_protocol"]
        else:
            _expect(
                episode_inference["inference_protocol"] == shared_inference_protocol,
                f"seed {seed_layout.seed} changes the shared inference protocol",
            )
        _expect(
            episode_training["protocol"]["image_min_pixels"]
            == episode_inference["inference_protocol"]["image_min_pixels"]
            and episode_training["protocol"]["image_max_pixels"]
            == episode_inference["inference_protocol"]["image_max_pixels"],
            f"seed {seed_layout.seed} train/inference image resolution differs",
        )
        seed_key = str(seed_layout.seed)
        seed_runs[seed_key] = {
            "seed": seed_layout.seed,
            "episode": {
                **episode_inference,
                "training": episode_training,
            },
            "isolated": {
                **isolated_inference,
                "training": isolated_training,
            },
            "episode_minus_isolated": _metric_delta(
                episode_inference, isolated_inference
            ),
            "scene_bootstrap": {
                "record_accuracy": _scene_bootstrap(
                    benchmark_rows=benchmark_rows,
                    record_ids=all_record_ids,
                    episode_correctness=episode_inference["record_correctness"],
                    isolated_correctness=isolated_inference["record_correctness"],
                    seed=layout.bootstrap_seed + seed_layout.seed * 2,
                    resamples=layout.bootstrap_resamples,
                ),
                "test_heldout_composition": _scene_bootstrap(
                    benchmark_rows=benchmark_rows,
                    record_ids=primary_record_ids,
                    episode_correctness=episode_inference["record_correctness"],
                    isolated_correctness=isolated_inference["record_correctness"],
                    seed=layout.bootstrap_seed + seed_layout.seed * 2 + 1,
                    resamples=layout.bootstrap_resamples,
                ),
            },
        }

    baselines: dict[str, Any] = {}
    for baseline_layout in layout.baselines:
        baseline = _collect_inference(
            name=baseline_layout.name,
            inference_dir=baseline_layout.inference_dir,
            benchmark=layout.benchmark,
            benchmark_sha=benchmark_sha,
            benchmark_rows=benchmark_rows,
            benchmark_ids=all_record_ids,
            composition_program_counts=composition_contract["program_counts"],
            composition_record_ids_by_program=composition_contract[
                "record_ids_by_program"
            ],
            model=layout.model,
            expected_model_inventory_sha256=model_inventory_sha256,
            expected_prompt_mode=layout.expected_prompt_mode,
            expected_image_mode=baseline_layout.image_mode,
            adapter_dir=None,
            expected_adapter_inventory_sha256=None,
        )
        _expect(
            baseline["inference_protocol"] == shared_inference_protocol,
            f"baseline {baseline_layout.name} changes the inference protocol",
        )
        baselines[baseline_layout.name] = baseline

    profile = matching["token_profile"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_status": {
            "status": "complete",
            "validated_seed_pairs": len(seed_runs),
            "validated_baselines": len(baselines),
        },
        "causal_validity": {
            "status": "valid_for_paired_groupstep_pilot",
            "valid": True,
            "optimizer_unit": "comparison_group",
            "claim_scope": "Qwen-VL group-step pilot comparison",
            "is_8b_main_result": False,
            "conditions": [
                "one optimizer update per comparison_id per epoch",
                "paired arms share facts, effective fact weights, and RGB image occurrences",
                "training and inference presentation contracts are shared",
            ],
        },
        "matching_scope": {
            "matched": [
                "fact identities and unit effective fact weights",
                "per-comparison RGB image-occurrence multisets",
                "processor-counted image tokens",
            ],
            "not_claimed_matched": [
                "text tokens",
                "total input tokens",
                "FLOPs",
                "wall-clock time",
            ],
            "comparison_groups": matching["comparison_groups"],
            "draws_per_arm": matching["draws_per_arm"],
            "facts": matching["facts"],
            "image_tokens": {
                "episode": profile["episode"].get("image_tokens"),
                "isolated": profile["isolated"].get("image_tokens"),
            },
            "total_input_tokens_observed_not_matched": {
                "episode": profile["episode"].get("input_tokens"),
                "isolated": profile["isolated"].get("input_tokens"),
            },
        },
        "contract": {
            "benchmark_records": len(benchmark_rows),
            "benchmark_sha256": benchmark_sha,
            "composition": composition_contract,
            "primary_endpoint": {
                "name": "test_only_heldout_composition",
                "split": PRIMARY_COMPOSITION_SPLIT,
                "records": len(primary_record_ids),
                "program_counts": layout.expected_primary_composition_program_counts,
                "record_ids": sorted(primary_record_ids),
                "directional_hypothesis_frozen_before_prediction_review": True,
            },
            "presentation_contract": layout.expected_presentation_contract,
            "prompt_mode": layout.expected_prompt_mode,
            "seeds": [item.seed for item in layout.seeds],
            "training_protocol": shared_training_protocol,
            "inference_protocol": shared_inference_protocol,
        },
        "provenance": {
            "config": _artifact(layout.config_path),
            "benchmark": _artifact(layout.benchmark),
            "composition_benchmark": _artifact(layout.composition_benchmark),
            "schedule_manifest": _artifact(layout.schedule_manifest),
            "token_profile": _artifact(layout.token_profile),
            "episode_schedule": _artifact(layout.episode_schedule),
            "isolated_schedule": _artifact(layout.isolated_schedule),
            "model": current_model_inventory,
        },
        "baselines": baselines,
        "seed_runs": seed_runs,
        "cross_seed": _cross_seed_summary(seed_runs),
        "uncertainty_scope": {
            "scene_bootstrap": "within-seed benchmark-scene sampling uncertainty",
            "cross_seed": "mean and observed range; seeds are not pooled as independent records",
        },
    }


def summarize_groupstep_pilot(layout: GroupStepPilotLayout) -> dict[str, Any]:
    """Validate and summarize a formal paired group-step pilot."""

    try:
        return _formal_summary(layout)
    except GroupStepSummaryError:
        raise
    except (PilotSummaryError, TrainingContractError, OSError, ValueError) as exc:
        raise GroupStepSummaryError(str(exc)) from exc


def _format_percent(value: Any) -> str:
    return "NA" if value is None else f"{float(value) * 100:.2f}%"


def _format_mean_range(summary: dict[str, Any]) -> str:
    if summary["mean"] is None:
        return "NA"
    return (
        f"{_format_percent(summary['mean'])} "
        f"[{_format_percent(summary['min'])}, {_format_percent(summary['max'])}]"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a concise human-readable companion to the lossless JSON report."""

    artifact = summary["artifact_status"]
    validity = summary["causal_validity"]
    primary_counts = summary["contract"]["primary_endpoint"]["program_counts"]
    primary_description = " + ".join(
        f"{count} `{program}`" for program, count in primary_counts.items()
    )
    lines = [
        "# EpiSpace formal group-step pilot",
        "",
        f"Artifact status: **{artifact['status']}**. Causal validity: "
        f"**{validity['status']}**.",
        "",
        "> Scope: paired Qwen-VL group-step pilot. This is not the 8B main result.",
        "> Matching claims are limited to facts and visual exposure; total tokens, FLOPs,",
        "> and wall-clock compute are not claimed matched.",
        "",
        "## Per-seed results",
        "",
        f"Primary endpoint is test-only held-out composition: {primary_description}.",
        "",
        "| Seed | Arm | Test composition | Test family | Test reveal | Test triple | Test ACES | Test parse |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, run in summary["seed_runs"].items():
        for arm in ("episode", "isolated"):
            values = _scalar_metrics(run[arm])
            test_parse = run[arm]["metrics_by_split"]["test"]["parse_rate"][
                "accuracy"
            ]
            lines.append(
                f"| {seed} | {arm} | "
                f"{_format_percent(values['test_heldout_composition'])} | "
                f"{_format_percent(values['test_family_exact_match'])} | "
                f"{_format_percent(values['test_revealed_accuracy'])} | "
                f"{_format_percent(values['test_evidence_triple_exact_match'])} | "
                f"{_format_percent(values['test_aces'])} | "
                f"{_format_percent(test_parse)} |"
            )
    lines.extend(
        [
            "",
            "### Primary endpoint by held-out signature",
            "",
            "| Seed | Arm | Program | Correct | Total | Accuracy |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for seed, run in summary["seed_runs"].items():
        for arm in ("episode", "isolated"):
            programs = run[arm]["metrics_by_split"]["test"]["per_program"]
            for program in primary_counts:
                metric = programs[program]
                lines.append(
                    f"| {seed} | {arm} | `{program}` | {metric['correct']} | "
                    f"{metric['total']} | {_format_percent(metric['accuracy'])} |"
                )
    lines.extend(
        [
            "",
            "## Cross-seed mean and observed range",
            "",
            "Values are mean [minimum, maximum] over training seeds; predictions are not pooled.",
            "",
            "| Metric | Episode | Isolated | Episode − isolated |",
            "|---|---:|---:|---:|",
        ]
    )
    labels = {
        "test_heldout_composition": "Test composition (primary)",
        "test_record_accuracy": "Test record",
        "test_family_exact_match": "Test family exact",
        "test_claim_pair_exact_match": "Test claim pair",
        "test_frame_pair_exact_match": "Test frame pair",
        "test_revealed_accuracy": "Test reveal",
        "test_evidence_triple_exact_match": "Test evidence triple",
        "test_aces": "Test ACES",
        "val_heldout_composition": "Val composition",
        "val_record_accuracy": "Val record",
        "val_family_exact_match": "Val family exact",
        "val_claim_pair_exact_match": "Val claim pair",
        "val_frame_pair_exact_match": "Val frame pair",
        "val_revealed_accuracy": "Val reveal",
        "val_evidence_triple_exact_match": "Val evidence triple",
        "val_aces": "Val ACES",
    }
    for metric, label in labels.items():
        lines.append(
            f"| {label} | {_format_mean_range(summary['cross_seed']['episode'][metric])} | "
            f"{_format_mean_range(summary['cross_seed']['isolated'][metric])} | "
            f"{_format_mean_range(summary['cross_seed']['episode_minus_isolated'][metric])} |"
        )
    lines.extend(
        [
            "",
            "### Cross-seed primary signatures",
            "",
            "| Program | Episode | Isolated | Episode − isolated |",
            "|---|---:|---:|---:|",
        ]
    )
    for program in primary_counts:
        program_summary = summary["cross_seed"]["per_split_program"]["test"][program]
        lines.append(
            f"| `{program}` | {_format_mean_range(program_summary['episode'])} | "
            f"{_format_mean_range(program_summary['isolated'])} | "
            f"{_format_mean_range(program_summary['episode_minus_isolated'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired scene-cluster bootstrap",
            "",
            "These intervals quantify benchmark-scene sampling uncertainty within each seed; "
            "they do not replace training-seed variation.",
            "",
            "| Seed | Endpoint | Delta | 95% CI | Scenes | Records |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for seed, run in summary["seed_runs"].items():
        for endpoint, label in (
            ("record_accuracy", "Record"),
            ("test_heldout_composition", "Test composition"),
        ):
            bootstrap = run["scene_bootstrap"][endpoint]
            low, high = bootstrap["ci95_percentile"]
            lines.append(
                f"| {seed} | {label} | {_format_percent(bootstrap['observed_delta'])} | "
                f"[{_format_percent(low)}, {_format_percent(high)}] | "
                f"{bootstrap['scene_clusters']} | {bootstrap['records']} |"
            )
    matching = summary["matching_scope"]
    lines.extend(
        [
            "",
            "## Verified matching scope",
            "",
            f"- Comparison groups: {matching['comparison_groups']}; facts: "
            f"{matching['facts']}; draws per arm: {matching['draws_per_arm']}.",
            f"- Image tokens: episode {matching['image_tokens']['episode']}, isolated "
            f"{matching['image_tokens']['isolated']}.",
            "- Not claimed matched: text tokens, total input tokens, FLOPs, or wall-clock time.",
            "",
            "## Provenance anchors",
            "",
            f"- Benchmark: `{summary['provenance']['benchmark']['sha256']}`",
            f"- Schedule manifest: `{summary['provenance']['schedule_manifest']['sha256']}`",
            f"- Token profile: `{summary['provenance']['token_profile']['sha256']}`",
            f"- Episode schedule: `{summary['provenance']['episode_schedule']['sha256']}`",
            f"- Isolated schedule: `{summary['provenance']['isolated_schedule']['sha256']}`",
            "",
            "The JSON companion contains every run manifest, adapter inventory, evaluator "
            "binding, per-seed metric, and bootstrap result.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary(summary: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    """Write the lossless JSON report and its Markdown companion."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

"""Fail-closed provenance and metric summary for the paired Qwen3-VL pilot.

The summary is deliberately stricter than a convenience plotting script.  It
refuses stale evaluator reports, mismatched benchmarks/schedules, incomplete
prediction coverage, and adapters that cannot be tied back to a completed
training run.  ``allow_incomplete`` only permits genuinely absent run
artifacts; it never turns an integrity error into a warning.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "epispace.qwen3vl_pilot_summary.v1"
INVENTORY_ARTIFACT_SCHEMA_VERSION = "epispace.weight_inventory.v1"
RUN_ORDER = ("base_full", "base_no_image", "episode_sft", "isolated_sft")
HELDOUT_COMPOSITION_SIGNATURES = {
    "counterfactual_cross_view.v1": (
        "G(view_a,a)+G(view_b,b)->F*_cross_view->B_global->R_canonical->V_claim"
    ),
    "target_view_prediction.v1": "B_global->F_query(origin,facing)->P_visibility(target)->V_render",
}
BOOTSTRAP_SEED = 1701
BOOTSTRAP_RESAMPLES = 10_000
METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "record_accuracy": ("record_accuracy",),
    "family_exact_match": ("family_exact_match",),
    "claim_pair_exact_match": ("claim_pair_exact_match",),
    "frame_pair_exact_match": ("frame_equivariance", "frame_pair_exact_match"),
    "revealed_accuracy": ("evidence_triples", "revealed_accuracy"),
    "evidence_triple_exact_match": ("evidence_triples", "exact_match"),
}
AGGREGATE_METRIC_ORDER = (
    "record_accuracy",
    "heldout_composition",
    "family_exact_match",
    "claim_pair_exact_match",
    "frame_pair_exact_match",
    "revealed_accuracy",
    "evidence_triple_exact_match",
    "aces",
)


class PilotSummaryError(RuntimeError):
    """Raised when a pilot artifact is absent, stale, or internally inconsistent."""


@dataclass(frozen=True)
class PilotLayout:
    """Filesystem layout for one paired pilot experiment."""

    experiment_root: Path
    benchmark: Path
    composition_benchmark: Path
    model: Path
    schedule_manifest: Path
    token_profile: Path
    episode_schedule: Path
    isolated_schedule: Path
    base_full_dir: Path
    base_no_image_dir: Path
    episode_train_dir: Path
    isolated_train_dir: Path

    @classmethod
    def defaults(
        cls,
        experiment_root: Path,
        benchmark: Path,
        model: Path,
        composition_benchmark: Path | None = None,
    ) -> PilotLayout:
        root = experiment_root.resolve()
        schedules = root / "pilot128_schedules"
        return cls(
            experiment_root=root,
            benchmark=benchmark.resolve(),
            composition_benchmark=(
                composition_benchmark.resolve()
                if composition_benchmark is not None
                else benchmark.resolve().with_name("benchmark.composition.jsonl")
            ),
            model=model.resolve(),
            schedule_manifest=schedules / "manifest.json",
            token_profile=schedules / "qwen_token_profile_256.json",
            episode_schedule=schedules / "image_matched.episode.schedule.jsonl",
            isolated_schedule=schedules / "image_matched.isolated.schedule.jsonl",
            base_full_dir=root / "base_256",
            base_no_image_dir=root / "base_no_image",
            episode_train_dir=root / "pilot128_seed17_episode",
            isolated_train_dir=root / "pilot128_seed17_isolated",
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PilotSummaryError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PilotSummaryError(f"required file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotSummaryError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotSummaryError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PilotSummaryError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PilotSummaryError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise PilotSummaryError(f"expected an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise PilotSummaryError(message)


def _program_id(row: dict[str, Any], source: str) -> str:
    program = row.get("program")
    _expect(isinstance(program, dict), f"{source} row has no program object")
    program_id = program.get("program_id")
    _expect(isinstance(program_id, str) and program_id, f"{source} row has no program_id")
    return program_id


def validate_composition_benchmark(
    benchmark_rows: list[dict[str, Any]],
    composition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove that the frozen composition file is the preregistered core subset."""

    _expect(bool(composition_rows), "composition benchmark is empty")
    expected_programs = set(HELDOUT_COMPOSITION_SIGNATURES)
    composition_by_id: dict[str, dict[str, Any]] = {}
    program_counts = {program_id: 0 for program_id in expected_programs}
    record_ids_by_program: dict[str, list[str]] = {
        program_id: [] for program_id in expected_programs
    }
    for row in composition_rows:
        record_id = row.get("record_id")
        _expect(
            isinstance(record_id, str) and record_id,
            "composition benchmark contains an invalid record_id",
        )
        _expect(
            record_id not in composition_by_id, "composition benchmark has duplicate record IDs"
        )
        program_id = _program_id(row, "composition benchmark")
        _expect(
            program_id in expected_programs,
            f"composition benchmark contains non-preregistered program {program_id}",
        )
        signature = row["program"].get("semantic_signature")
        _expect(
            signature == HELDOUT_COMPOSITION_SIGNATURES[program_id],
            f"composition benchmark signature mismatch for {program_id}",
        )
        composition_by_id[record_id] = row
        program_counts[program_id] += 1
        record_ids_by_program[program_id].append(record_id)
    _expect(
        all(count > 0 for count in program_counts.values()),
        "composition benchmark does not cover both preregistered programs",
    )

    core_subset: dict[str, dict[str, Any]] = {}
    for row in benchmark_rows:
        program_id = _program_id(row, "core benchmark")
        if program_id in expected_programs:
            record_id = row.get("record_id")
            _expect(
                isinstance(record_id, str) and record_id, "core benchmark has invalid record_id"
            )
            core_subset[record_id] = row
    _expect(
        set(core_subset) == set(composition_by_id),
        "composition benchmark IDs do not exactly match the held-out-program core subset",
    )
    for record_id, row in composition_by_id.items():
        _expect(
            row == core_subset[record_id],
            f"composition/core payload mismatch for {record_id}",
        )
    return {
        "records": len(composition_rows),
        "program_ids": sorted(expected_programs),
        "program_counts": dict(sorted(program_counts.items())),
        "record_ids_by_program": {
            key: sorted(value) for key, value in sorted(record_ids_by_program.items())
        },
        "semantic_signatures": dict(sorted(HELDOUT_COMPOSITION_SIGNATURES.items())),
        "record_ids": sorted(composition_by_id),
    }


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _inventory(files: list[Path], root: Path) -> dict[str, Any]:
    unique = sorted({path.resolve() for path in files}, key=lambda path: str(path))
    _expect(bool(unique), f"artifact inventory is empty under {root}")
    entries = []
    file_stats: list[dict[str, Any]] = []
    root_resolved = root.resolve()
    for path in unique:
        _expect(path.is_file(), f"inventory file is missing: {path}")
        try:
            relative = path.relative_to(root_resolved).as_posix()
        except ValueError as exc:
            raise PilotSummaryError(f"inventory file escapes root {root}: {path}") from exc
        before = _stat_identity(path)
        digest = sha256(path)
        after = _stat_identity(path)
        _expect(
            before == after,
            f"inventory file changed while it was being hashed: {path}",
        )
        entries.append(
            {
                "path": relative,
                "bytes": after["bytes"],
                "sha256": digest,
            }
        )
        file_stats.append({"path": relative, **after})
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "root": str(root_resolved),
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "file_stats": file_stats,
    }


def _model_runtime_role(relative: str) -> str | None:
    """Classify files consumed by ``from_pretrained`` at runtime."""

    path = Path(relative)
    name = path.name
    exact_roles = {
        "generation_config.json": "generation_config",
        "preprocessor_config.json": "image_preprocessor_config",
        "video_preprocessor_config.json": "video_preprocessor_config",
        "processor_config.json": "processor_config",
        "processing_config.json": "processor_config",
        "tokenizer_config.json": "tokenizer_config",
        "tokenizer.json": "tokenizer",
        "tokenizer.model": "tokenizer_model",
        "spiece.model": "tokenizer_model",
        "sentencepiece.bpe.model": "tokenizer_model",
        "vocab.json": "tokenizer_vocab",
        "vocab.txt": "tokenizer_vocab",
        "merges.txt": "tokenizer_merges",
        "special_tokens_map.json": "tokenizer_special_tokens",
        "added_tokens.json": "tokenizer_added_tokens",
    }
    if name in exact_roles:
        return exact_roles[name]
    if name.startswith("chat_template") and path.suffix in {".json", ".jinja"}:
        return "chat_template"
    if path.parts[:1] == ("chat_templates",) and path.suffix in {".json", ".jinja"}:
        return "chat_template"
    if name.startswith("tokenizer") and path.suffix in {".json", ".model"}:
        return "tokenizer"
    if name.startswith("vocab.") and path.suffix in {".json", ".txt"}:
        return "tokenizer_vocab"
    if name.startswith("merges.") and path.suffix in {".json", ".txt"}:
        return "tokenizer_merges"
    return None


def _model_files_and_roles(model_dir: Path) -> tuple[list[Path], dict[str, str]]:
    """Resolve weights and processor/tokenizer runtime state without hashing."""

    model_dir = model_dir.resolve()
    config = model_dir / "config.json"
    _expect(config.is_file(), f"model config is missing: {config}")
    index_candidates = sorted(model_dir.glob("*.index.json"))
    files = [config]
    roles: dict[str, str] = {"config.json": "config"}
    if index_candidates:
        _expect(
            len(index_candidates) == 1,
            f"expected one model weight index under {model_dir}, found {len(index_candidates)}",
        )
        index = index_candidates[0]
        index_data = _load_json(index)
        weight_map = index_data.get("weight_map")
        _expect(isinstance(weight_map, dict) and weight_map, f"invalid weight_map in {index}")
        _expect(
            all(isinstance(name, str) and name for name in weight_map.values()),
            f"non-string weight shard in {index}",
        )
        shard_names = sorted(set(weight_map.values()))
        for name in shard_names:
            shard = (model_dir / name).resolve()
            try:
                shard.relative_to(model_dir)
            except ValueError as exc:
                raise PilotSummaryError(
                    f"weight index path escapes model directory: {name}"
                ) from exc
            _expect(shard.is_file(), f"weight shard referenced by index is missing: {shard}")
            files.append(shard)
            roles[name] = "weight"
        files.append(index)
        roles[index.name] = "weight_index"
    else:
        weights = sorted(
            {
                *model_dir.glob("*.safetensors"),
                *model_dir.glob("pytorch_model*.bin"),
                *model_dir.glob("*.pt"),
            }
        )
        _expect(bool(weights), f"no model weight files found under {model_dir}")
        files.extend(weights)
        roles.update({path.name: "weight" for path in weights})
    for runtime_file in sorted(path for path in model_dir.rglob("*") if path.is_file()):
        relative = runtime_file.resolve().relative_to(model_dir).as_posix()
        role = _model_runtime_role(relative)
        if role is None or relative in roles:
            continue
        files.append(runtime_file)
        roles[relative] = role
    return files, roles


def model_inventory(model_dir: Path) -> dict[str, Any]:
    """Hash weights plus all processor/tokenizer state used during inference."""

    model_dir = model_dir.resolve()
    files, roles = _model_files_and_roles(model_dir)
    result = _inventory(files, model_dir)
    for entry in result["files"]:
        entry["role"] = roles[entry["path"]]
    return result


def directory_inventory(directory: Path) -> dict[str, Any]:
    _expect(directory.is_dir(), f"required directory is missing: {directory}")
    return _inventory([path for path in directory.rglob("*") if path.is_file()], directory)


def _canonical_inventory_entries(entries: Any, root: Path) -> list[dict[str, Any]]:
    _expect(isinstance(entries, list) and entries, f"inventory is empty under {root}")
    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        _expect(isinstance(raw, dict), f"inventory entry {index} is not an object")
        relative = raw.get("path")
        size = raw.get("bytes")
        digest = raw.get("sha256")
        _expect(
            isinstance(relative, str) and relative and relative not in seen,
            f"inventory entry {index} has an invalid or duplicate path",
        )
        _expect(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"inventory entry {relative} has invalid bytes",
        )
        _expect(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"inventory entry {relative} has invalid SHA-256",
        )
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PilotSummaryError(
                f"inventory entry escapes root {root}: {relative}"
            ) from exc
        seen.add(relative)
        canonical.append({"path": relative, "bytes": size, "sha256": digest})
    return sorted(canonical, key=lambda entry: entry["path"])


def write_inventory_artifact(
    root: Path, output: Path, *, kind: str, overwrite: bool = False
) -> dict[str, Any]:
    """Materialize one content-hashed inventory for reuse across GPU stages.

    Large model shards are read only here (and once again by the final formal
    reporter).  Consumers validate the artifact plus an exact filesystem stat
    snapshot, so a bare, unverified digest supplied on the command line is
    never accepted.
    """

    root = root.resolve()
    output = output.resolve()
    _expect(kind in {"model", "directory"}, f"unsupported inventory kind: {kind}")
    if output.exists() and not overwrite:
        return verify_inventory_artifact(output, root=root, kind=kind)
    inventory = model_inventory(root) if kind == "model" else directory_inventory(root)
    payload = {
        "schema_version": INVENTORY_ARTIFACT_SCHEMA_VERSION,
        "kind": kind,
        "root": str(root),
        "inventory": inventory,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return verify_inventory_artifact(output, root=root, kind=kind)


def verify_inventory_artifact(
    artifact: Path,
    *,
    root: Path,
    kind: str,
    verify_content: bool = False,
) -> dict[str, Any]:
    """Validate a reusable inventory artifact and return its manifest binding."""

    artifact = artifact.resolve()
    root = root.resolve()
    payload = _load_json(artifact)
    _expect(
        payload.get("schema_version") == INVENTORY_ARTIFACT_SCHEMA_VERSION,
        f"unsupported inventory artifact schema: {artifact}",
    )
    _expect(payload.get("kind") == kind, f"inventory artifact kind mismatch: {artifact}")
    _expect(
        Path(str(payload.get("root", ""))).resolve() == root,
        f"inventory artifact root mismatch: {artifact}",
    )
    inventory = payload.get("inventory")
    _expect(isinstance(inventory, dict), f"inventory artifact has no inventory: {artifact}")
    _expect(
        Path(str(inventory.get("root", ""))).resolve() == root,
        f"embedded inventory root mismatch: {artifact}",
    )
    entries = _canonical_inventory_entries(inventory.get("files"), root)
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    inventory_digest = hashlib.sha256(canonical).hexdigest()
    _expect(
        inventory.get("inventory_sha256") == inventory_digest,
        f"inventory aggregate digest mismatch: {artifact}",
    )
    _expect(inventory.get("file_count") == len(entries), "inventory file count mismatch")
    _expect(
        inventory.get("total_bytes") == sum(entry["bytes"] for entry in entries),
        f"inventory byte count mismatch: {artifact}",
    )
    if kind == "model":
        expected_files, expected_roles = _model_files_and_roles(root)
        expected_paths = {
            path.resolve().relative_to(root).as_posix() for path in expected_files
        }
        _expect(
            {entry["path"] for entry in entries} == expected_paths,
            f"model inventory does not cover the resolved runtime file set: {artifact}",
        )
        by_path = {
            str(raw["path"]): raw for raw in inventory["files"] if isinstance(raw, dict)
        }
        _expect(
            all(by_path[path].get("role") == expected_roles[path] for path in expected_paths),
            f"model inventory roles are invalid: {artifact}",
        )
    else:
        current_paths = {
            path.resolve().relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        _expect(
            {entry["path"] for entry in entries} == current_paths,
            f"directory inventory coverage changed: {artifact}",
        )
    raw_stats = inventory.get("file_stats")
    _expect(isinstance(raw_stats, list), f"inventory stat snapshot is missing: {artifact}")
    stats_by_path = {
        str(item.get("path")): item for item in raw_stats if isinstance(item, dict)
    }
    _expect(
        set(stats_by_path) == {entry["path"] for entry in entries},
        f"inventory stat snapshot coverage mismatch: {artifact}",
    )
    for entry in entries:
        current = _stat_identity(root / entry["path"])
        expected = stats_by_path[entry["path"]]
        _expect(
            all(expected.get(key) == value for key, value in current.items()),
            f"inventory file identity changed: {root / entry['path']}",
        )
        _expect(current["bytes"] == entry["bytes"], "inventory file size changed")
    if verify_content:
        observed = model_inventory(root) if kind == "model" else directory_inventory(root)
        _expect(
            observed["inventory_sha256"] == inventory_digest,
            f"inventory content digest is stale: {artifact}",
        )
    return {
        "kind": kind,
        "root": str(root),
        "inventory_artifact": str(artifact),
        "inventory_artifact_sha256": sha256(artifact),
        "inventory_sha256": inventory_digest,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "verification": (
            "content_sha256_recomputed" if verify_content else "artifact_structure_and_stat_identity"
        ),
    }


def validate_inventory_binding(
    binding: Any,
    *,
    root: Path,
    kind: str,
    expected_inventory_sha256: str | None = None,
    verify_content: bool = False,
) -> dict[str, Any]:
    """Revalidate a manifest's inventory binding instead of trusting its digest."""

    _expect(isinstance(binding, dict), f"missing {kind} inventory binding")
    artifact_value = binding.get("inventory_artifact")
    _expect(
        isinstance(artifact_value, str) and artifact_value,
        f"{kind} inventory binding has no artifact path",
    )
    observed = verify_inventory_artifact(
        Path(artifact_value), root=root, kind=kind, verify_content=verify_content
    )
    for key in (
        "kind",
        "root",
        "inventory_artifact",
        "inventory_artifact_sha256",
        "inventory_sha256",
        "file_count",
        "total_bytes",
    ):
        _expect(
            binding.get(key) == observed[key],
            f"{kind} inventory binding mismatch for {key}",
        )
    if expected_inventory_sha256 is not None:
        _expect(
            observed["inventory_sha256"] == expected_inventory_sha256,
            f"{kind} inventory digest differs across stages",
        )
    return observed


def _metric_at(report: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value: Any = report
    for key in path:
        _expect(isinstance(value, dict) and key in value, f"evaluation is missing {'.'.join(path)}")
        value = value[key]
    _expect(isinstance(value, dict), f"evaluation metric {'.'.join(path)} is not an object")
    for key in ("correct", "total", "accuracy", "status"):
        _expect(key in value, f"evaluation metric {'.'.join(path)} is missing {key}")
    accuracy = value["accuracy"]
    correct = value["correct"]
    total = value["total"]
    _expect(
        isinstance(correct, int)
        and not isinstance(correct, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and 0 <= correct <= total,
        f"evaluation metric {'.'.join(path)} has invalid counts",
    )
    _expect(
        accuracy is None or isinstance(accuracy, int | float),
        f"evaluation metric {'.'.join(path)} has invalid accuracy",
    )
    expected_accuracy = correct / total if total else None
    _expect(
        (accuracy is None and expected_accuracy is None)
        or (
            isinstance(accuracy, int | float)
            and math.isfinite(float(accuracy))
            and expected_accuracy is not None
            and math.isclose(float(accuracy), expected_accuracy, rel_tol=0.0, abs_tol=1e-12)
        ),
        f"evaluation metric {'.'.join(path)} accuracy/count inconsistency",
    )
    return {key: value[key] for key in ("correct", "total", "accuracy", "status")}


def extract_metrics(
    report: dict[str, Any], composition_program_counts: dict[str, int]
) -> dict[str, Any]:
    metrics = {name: _metric_at(report, path) for name, path in METRIC_PATHS.items()}
    try:
        aces = report["evidence_triples"]["accuracy_conditioned_evidence_sensitivity"]
    except (KeyError, TypeError) as exc:
        raise PilotSummaryError("evaluation is missing evidence sensitivity (ACES)") from exc
    _expect(isinstance(aces, dict), "ACES metric is not an object")
    for key in ("numerator_joint_correct", "denominator_revealed_correct", "value", "status"):
        _expect(key in aces, f"ACES metric is missing {key}")
    metrics["aces"] = {
        key: aces[key]
        for key in ("numerator_joint_correct", "denominator_revealed_correct", "value", "status")
    }
    numerator = metrics["aces"]["numerator_joint_correct"]
    denominator = metrics["aces"]["denominator_revealed_correct"]
    value = metrics["aces"]["value"]
    _expect(
        isinstance(numerator, int)
        and not isinstance(numerator, bool)
        and isinstance(denominator, int)
        and not isinstance(denominator, bool)
        and 0 <= numerator <= denominator,
        "ACES has invalid counts",
    )
    expected_aces = numerator / denominator if denominator else None
    _expect(
        (value is None and expected_aces is None)
        or (
            isinstance(value, int | float)
            and math.isfinite(float(value))
            and expected_aces is not None
            and math.isclose(float(value), expected_aces, rel_tol=0.0, abs_tol=1e-12)
        ),
        "ACES value/count inconsistency",
    )
    per_program = report.get("per_program")
    _expect(isinstance(per_program, dict) and per_program, "evaluation has no per_program metrics")
    metrics["per_program"] = {
        str(name): _metric_at({"metric": metric}, ("metric",))
        for name, metric in sorted(per_program.items())
    }
    heldout_correct = 0
    heldout_total = 0
    for program_id, expected_total in composition_program_counts.items():
        _expect(
            program_id in metrics["per_program"],
            f"evaluation is missing held-out composition program {program_id}",
        )
        metric = metrics["per_program"][program_id]
        _expect(
            metric["total"] == expected_total,
            f"evaluation total for {program_id} does not match composition benchmark",
        )
        heldout_correct += metric["correct"]
        heldout_total += metric["total"]
    expected_composition_total = sum(composition_program_counts.values())
    _expect(
        heldout_total == expected_composition_total,
        "held-out composition aggregate denominator mismatch",
    )
    metrics["heldout_composition"] = {
        "correct": heldout_correct,
        "total": heldout_total,
        "accuracy": heldout_correct / heldout_total,
        "status": "defined",
        "program_ids": sorted(composition_program_counts),
        "preregistered_primary_endpoint": True,
    }
    return metrics


def _required_run_files(inference_dir: Path, train_dir: Path | None) -> list[Path]:
    files = [
        inference_dir / "prediction_manifest.json",
        inference_dir / "predictions.jsonl",
        inference_dir / "raw_generations.jsonl",
        inference_dir / "evaluation" / "evaluation.json",
        inference_dir / "evaluation" / "evaluation.md",
    ]
    if train_dir is not None:
        files.extend([train_dir / "run_manifest.json", train_dir / "train_log.jsonl"])
        files.append(train_dir / "adapter")
    return files


def _collect_run(
    *,
    name: str,
    inference_dir: Path,
    train_dir: Path | None,
    expected_image_mode: str,
    benchmark: Path,
    benchmark_sha: str,
    benchmark_records: int,
    benchmark_ids: set[str],
    composition_program_counts: dict[str, int],
    composition_record_ids_by_program: dict[str, list[str]],
    model: Path,
    expected_schedule: Path | None,
    expected_schedule_sha: str | None,
    allow_incomplete: bool,
) -> dict[str, Any]:
    required_inference = _required_run_files(inference_dir, None)
    missing = [
        str(path) for path in _required_run_files(inference_dir, train_dir) if not path.exists()
    ]
    if missing:
        if allow_incomplete:
            _expect(
                not any(path.exists() for path in required_inference),
                f"run {name} has a partially materialized inference; refusing an ambiguous snapshot",
            )
            return {"status": "missing", "missing_artifacts": missing}
        raise PilotSummaryError(f"run {name} is incomplete; missing: " + ", ".join(missing))

    prediction_manifest_path = inference_dir / "prediction_manifest.json"
    predictions_path = inference_dir / "predictions.jsonl"
    raw_path = inference_dir / "raw_generations.jsonl"
    evaluation_path = inference_dir / "evaluation" / "evaluation.json"
    evaluation_md_path = inference_dir / "evaluation" / "evaluation.md"
    prediction_manifest = _load_json(prediction_manifest_path)
    evaluation = _load_json(evaluation_path)
    predictions = _read_jsonl(predictions_path)
    raw = _read_jsonl(raw_path)

    _expect(
        prediction_manifest.get("schema_version") == "epispace.qwen3vl_predictions.v1",
        f"run {name} has unsupported prediction manifest schema",
    )
    _expect(
        Path(str(prediction_manifest.get("model", ""))).resolve() == model.resolve(),
        f"run {name} uses a different base model",
    )
    _expect(
        prediction_manifest.get("benchmark_sha256") == benchmark_sha,
        f"run {name} prediction manifest benchmark hash mismatch",
    )
    declared_benchmark = Path(str(prediction_manifest.get("benchmark", ""))).resolve()
    _expect(declared_benchmark == benchmark.resolve(), f"run {name} benchmark path mismatch")
    # image_mode was added after the first full-image pilot was launched.  The
    # v1 manifest's omitted value has exactly one defined legacy meaning: full.
    observed_image_mode = prediction_manifest.get("image_mode", "full")
    _expect(observed_image_mode == expected_image_mode, f"run {name} image mode mismatch")
    _expect(prediction_manifest.get("num_shards") == 1, f"run {name} is not a merged run")
    _expect(
        prediction_manifest.get("records") == benchmark_records, f"run {name} record count mismatch"
    )
    _expect(len(predictions) == benchmark_records, f"run {name} predictions JSONL count mismatch")
    _expect(len(raw) == benchmark_records, f"run {name} raw generations count mismatch")
    prediction_ids = [row.get("record_id") for row in predictions]
    raw_ids = [row.get("record_id") for row in raw]
    _expect(
        len(set(prediction_ids)) == benchmark_records, f"run {name} predictions have duplicate IDs"
    )
    _expect(
        set(prediction_ids) == benchmark_ids,
        f"run {name} prediction record IDs do not exactly cover the benchmark",
    )
    _expect(set(prediction_ids) == set(raw_ids), f"run {name} raw/prediction record IDs differ")
    _expect(
        Path(str(prediction_manifest.get("predictions", ""))).resolve()
        == predictions_path.resolve(),
        f"run {name} prediction manifest points to a different predictions file",
    )
    _expect(
        Path(str(prediction_manifest.get("raw_generations", ""))).resolve() == raw_path.resolve(),
        f"run {name} prediction manifest points to a different raw generations file",
    )

    _expect(
        evaluation.get("schema_version")
        in {
            "epispace.benchmark_evaluation.v1",  # legacy draw-step diagnostics
            "epispace.benchmark_evaluation.v2",
        },
        f"run {name} has unsupported evaluation schema",
    )
    inputs = evaluation.get("inputs")
    _expect(isinstance(inputs, dict), f"run {name} evaluation has no input binding")
    _expect(
        inputs.get("benchmark_sha256") == benchmark_sha, f"run {name} evaluator benchmark is stale"
    )
    predictions_sha = sha256(predictions_path)
    _expect(
        inputs.get("predictions_sha256") == predictions_sha,
        f"run {name} evaluator predictions are stale",
    )
    input_benchmark = Path(str(inputs.get("benchmark", ""))).resolve()
    input_predictions = Path(str(inputs.get("predictions", ""))).resolve()
    _expect(input_benchmark == benchmark.resolve(), f"run {name} evaluator benchmark path mismatch")
    _expect(
        input_predictions == predictions_path.resolve(),
        f"run {name} evaluator prediction path mismatch",
    )
    coverage = evaluation.get("predictions")
    _expect(isinstance(coverage, dict), f"run {name} evaluation has no coverage report")
    for key in ("missing_count", "extra_count", "duplicate_id_count"):
        _expect(coverage.get(key) == 0, f"run {name} evaluator reports {key}={coverage.get(key)!r}")
    _expect(
        evaluation.get("benchmark", {}).get("records") == benchmark_records,
        f"run {name} evaluator benchmark record count mismatch",
    )
    record_results = evaluation.get("record_results")
    _expect(isinstance(record_results, list), f"run {name} evaluation has no record_results")
    correctness: dict[str, bool] = {}
    for result in record_results:
        _expect(isinstance(result, dict), f"run {name} has a malformed record result")
        record_id = result.get("record_id")
        correct = result.get("correct")
        _expect(
            isinstance(record_id, str) and record_id in benchmark_ids,
            f"run {name} record_results contains an unknown ID",
        )
        _expect(record_id not in correctness, f"run {name} record_results has duplicate IDs")
        _expect(isinstance(correct, bool), f"run {name} record result correctness is not boolean")
        correctness[record_id] = correct
    _expect(
        set(correctness) == benchmark_ids,
        f"run {name} record_results do not exactly cover the benchmark",
    )
    _expect(
        sum(correctness.values()) == evaluation.get("record_accuracy", {}).get("correct"),
        f"run {name} record_results disagree with record accuracy",
    )

    artifacts: dict[str, Any] = {
        "prediction_manifest": _artifact(prediction_manifest_path),
        "predictions": _artifact(predictions_path),
        "raw_generations": _artifact(raw_path),
        "evaluation_json": _artifact(evaluation_path),
        "evaluation_markdown": _artifact(evaluation_md_path),
    }
    if train_dir is None:
        _expect(
            prediction_manifest.get("adapter") is None,
            f"base run {name} unexpectedly uses an adapter",
        )
    else:
        run_manifest_path = train_dir / "run_manifest.json"
        run_manifest = _load_json(run_manifest_path)
        _expect(
            run_manifest.get("schema_version") == "epispace.qwen3vl_lora_run.v1",
            f"run {name} has unsupported training manifest schema",
        )
        _expect(run_manifest.get("status") == "complete", f"run {name} training is not complete")
        config = run_manifest.get("config")
        _expect(isinstance(config, dict), f"run {name} has no training config")
        _expect(
            Path(str(config.get("model", ""))).resolve() == model.resolve(),
            f"run {name} training model mismatch",
        )
        _expect(
            expected_schedule is not None and expected_schedule_sha is not None,
            "internal schedule contract error",
        )
        _expect(
            Path(str(config.get("schedule", ""))).resolve() == expected_schedule.resolve(),
            f"run {name} training schedule path mismatch",
        )
        _expect(
            run_manifest.get("schedule_sha256") == expected_schedule_sha,
            f"run {name} training schedule is stale",
        )
        schedule_draws = len(_read_jsonl(expected_schedule))
        _expect(
            run_manifest.get("draws") == schedule_draws,
            f"run {name} training draw count does not match its schedule",
        )
        _expect(config.get("seed") == 17, f"run {name} is not the declared seed-17 pilot")
        _expect(
            config.get("max_steps") is None,
            f"run {name} used max_steps and is not a complete scheduled epoch run",
        )
        adapter_dir = train_dir / "adapter"
        declared_adapter = Path(str(prediction_manifest.get("adapter", ""))).resolve()
        _expect(declared_adapter == adapter_dir.resolve(), f"run {name} inference adapter mismatch")
        artifacts.update(
            {
                "training_manifest": _artifact(run_manifest_path),
                "training_log": _artifact(train_dir / "train_log.jsonl"),
                "adapter": directory_inventory(adapter_dir),
            }
        )

    metrics = extract_metrics(evaluation, composition_program_counts)
    for program_id, record_ids in composition_record_ids_by_program.items():
        expected_correct = sum(correctness[record_id] for record_id in record_ids)
        _expect(
            metrics["per_program"][program_id]["correct"] == expected_correct,
            f"run {name} per-program correctness disagrees with record_results for {program_id}",
        )

    return {
        "status": "complete",
        "image_mode": observed_image_mode,
        "records": benchmark_records,
        "parsed": prediction_manifest.get("parsed"),
        "inference_contract": {
            key: prediction_manifest.get(key)
            for key in ("image_min_pixels", "image_max_pixels", "max_new_tokens")
        },
        "metrics": metrics,
        "record_correctness": correctness,
        "artifacts": artifacts,
    }


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    values = {name: metric["accuracy"] for name, metric in metrics.items() if name in METRIC_PATHS}
    values["heldout_composition"] = metrics["heldout_composition"]["accuracy"]
    values["aces"] = metrics["aces"]["value"]
    return values


def _metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_values = _scalar_metrics(left)
    right_values = _scalar_metrics(right)
    overall = {
        key: (left_values[key] - right_values[key])
        if left_values[key] is not None and right_values[key] is not None
        else None
        for key in left_values
    }
    programs = sorted(set(left["per_program"]) | set(right["per_program"]))
    per_program = {}
    for program in programs:
        left_metric = left["per_program"].get(program)
        right_metric = right["per_program"].get(program)
        left_value = left_metric.get("accuracy") if left_metric else None
        right_value = right_metric.get("accuracy") if right_metric else None
        per_program[program] = (
            left_value - right_value if left_value is not None and right_value is not None else None
        )
    return {"overall": overall, "per_program": per_program}


def _validate_inference_contracts(runs: dict[str, Any]) -> None:
    full_runs = [runs[name] for name in ("base_full", "episode_sft", "isolated_sft")]
    contracts = [run["inference_contract"] for run in full_runs if run["status"] == "complete"]
    if contracts:
        _expect(
            all(contract == contracts[0] for contract in contracts),
            "full-image inference contracts differ",
        )


def _percentile(sorted_values: list[float], probability: float) -> float:
    _expect(bool(sorted_values), "cannot take percentile of an empty bootstrap sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _scene_cluster_bootstrap(
    *,
    benchmark_rows: list[dict[str, Any]],
    record_ids: set[str],
    left_correctness: dict[str, bool],
    right_correctness: dict[str, bool],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    _expect(resamples >= 10_000, "scene-cluster bootstrap requires at least 10k resamples")
    clusters: dict[str, list[str]] = {}
    for row in benchmark_rows:
        record_id = row.get("record_id")
        if record_id not in record_ids:
            continue
        scene_id = row.get("scene_id")
        _expect(
            isinstance(scene_id, str) and scene_id,
            f"bootstrap record {record_id} has no scene_id",
        )
        clusters.setdefault(scene_id, []).append(record_id)
    covered_ids = {record_id for ids in clusters.values() for record_id in ids}
    _expect(covered_ids == record_ids, "bootstrap scene clusters do not cover endpoint records")
    scene_ids = sorted(clusters)
    _expect(len(scene_ids) >= 2, "scene-cluster bootstrap needs at least two scenes")

    differences = {
        record_id: int(left_correctness[record_id]) - int(right_correctness[record_id])
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
        "observed_delta": observed,
        "ci95_percentile": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "records": len(record_ids),
        "scene_clusters": len(scene_ids),
        "seed": seed,
        "resamples": resamples,
    }


def bootstrap_uncertainty(
    runs: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
    composition_record_ids: set[str],
) -> dict[str, Any]:
    if any(runs[name]["status"] != "complete" for name in ("episode_sft", "isolated_sft")):
        return {
            "status": "unavailable_incomplete_runs",
            "contrast": "episode_sft_minus_isolated_sft",
        }
    all_record_ids = {str(row["record_id"]) for row in benchmark_rows}
    episode_correctness = runs["episode_sft"]["record_correctness"]
    isolated_correctness = runs["isolated_sft"]["record_correctness"]
    result = {
        "status": "defined",
        "contrast": "episode_sft_minus_isolated_sft",
        "method": "paired nonparametric percentile bootstrap over scene clusters",
        "interpretation": (
            "sampling uncertainty over benchmark scenes only; this does not replace "
            "training-seed variance"
        ),
        "record_accuracy": _scene_cluster_bootstrap(
            benchmark_rows=benchmark_rows,
            record_ids=all_record_ids,
            left_correctness=episode_correctness,
            right_correctness=isolated_correctness,
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        ),
        "heldout_composition": _scene_cluster_bootstrap(
            benchmark_rows=benchmark_rows,
            record_ids=composition_record_ids,
            left_correctness=episode_correctness,
            right_correctness=isolated_correctness,
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        ),
    }
    return result


def summarize_pilot(layout: PilotLayout, *, allow_incomplete: bool = False) -> dict[str, Any]:
    """Validate all available artifacts and return a lossless pilot summary."""

    shared_paths = [
        layout.benchmark,
        layout.composition_benchmark,
        layout.schedule_manifest,
        layout.token_profile,
        layout.episode_schedule,
        layout.isolated_schedule,
    ]
    missing_shared = [str(path) for path in shared_paths if not path.is_file()]
    _expect(not missing_shared, "shared pilot artifacts are missing: " + ", ".join(missing_shared))
    benchmark_rows = _read_jsonl(layout.benchmark)
    _expect(bool(benchmark_rows), "benchmark is empty")
    composition_rows = _read_jsonl(layout.composition_benchmark)
    composition_contract = validate_composition_benchmark(benchmark_rows, composition_rows)
    benchmark_ids = [row.get("record_id") for row in benchmark_rows]
    _expect(
        all(isinstance(record_id, str) and record_id for record_id in benchmark_ids),
        "benchmark contains an invalid record_id",
    )
    _expect(len(set(benchmark_ids)) == len(benchmark_rows), "benchmark record IDs are not unique")
    benchmark_sha = sha256(layout.benchmark)
    episode_schedule_sha = sha256(layout.episode_schedule)
    isolated_schedule_sha = sha256(layout.isolated_schedule)

    schedule_manifest = _load_json(layout.schedule_manifest)
    _expect(schedule_manifest.get("status") == "pass", "pilot schedule manifest status is not pass")
    artifacts = schedule_manifest.get("artifacts")
    _expect(isinstance(artifacts, dict), "schedule manifest has no artifacts binding")
    for arm, path, observed_sha in (
        ("episode", layout.episode_schedule, episode_schedule_sha),
        ("isolated", layout.isolated_schedule, isolated_schedule_sha),
    ):
        binding = artifacts.get(arm)
        _expect(isinstance(binding, dict), f"schedule manifest is missing {arm} binding")
        bound_path = (layout.schedule_manifest.parent / str(binding.get("path", ""))).resolve()
        _expect(bound_path == path.resolve(), f"schedule manifest {arm} path mismatch")
        _expect(binding.get("sha256") == observed_sha, f"schedule manifest {arm} hash mismatch")

    token_profile = _load_json(layout.token_profile)
    _expect(token_profile.get("status") == "pass", "token profile status is not pass")
    profile_sources = token_profile.get("sources")
    _expect(isinstance(profile_sources, dict), "token profile has no source bindings")
    _expect(
        profile_sources.get("episode_schedule_sha256") == episode_schedule_sha,
        "token profile episode schedule is stale",
    )
    _expect(
        profile_sources.get("isolated_schedule_sha256") == isolated_schedule_sha,
        "token profile isolated schedule is stale",
    )
    _expect(
        Path(str(token_profile.get("model", ""))).resolve() == layout.model.resolve(),
        "token profile model mismatch",
    )

    runs = {
        "base_full": _collect_run(
            name="base_full",
            inference_dir=layout.base_full_dir,
            train_dir=None,
            expected_image_mode="full",
            benchmark=layout.benchmark,
            benchmark_sha=benchmark_sha,
            benchmark_records=len(benchmark_rows),
            benchmark_ids=set(benchmark_ids),
            composition_program_counts=composition_contract["program_counts"],
            composition_record_ids_by_program=composition_contract["record_ids_by_program"],
            model=layout.model,
            expected_schedule=None,
            expected_schedule_sha=None,
            allow_incomplete=allow_incomplete,
        ),
        "base_no_image": _collect_run(
            name="base_no_image",
            inference_dir=layout.base_no_image_dir,
            train_dir=None,
            expected_image_mode="none",
            benchmark=layout.benchmark,
            benchmark_sha=benchmark_sha,
            benchmark_records=len(benchmark_rows),
            benchmark_ids=set(benchmark_ids),
            composition_program_counts=composition_contract["program_counts"],
            composition_record_ids_by_program=composition_contract["record_ids_by_program"],
            model=layout.model,
            expected_schedule=None,
            expected_schedule_sha=None,
            allow_incomplete=allow_incomplete,
        ),
        "episode_sft": _collect_run(
            name="episode_sft",
            inference_dir=layout.episode_train_dir / "inference_core_256",
            train_dir=layout.episode_train_dir,
            expected_image_mode="full",
            benchmark=layout.benchmark,
            benchmark_sha=benchmark_sha,
            benchmark_records=len(benchmark_rows),
            benchmark_ids=set(benchmark_ids),
            composition_program_counts=composition_contract["program_counts"],
            composition_record_ids_by_program=composition_contract["record_ids_by_program"],
            model=layout.model,
            expected_schedule=layout.episode_schedule,
            expected_schedule_sha=episode_schedule_sha,
            allow_incomplete=allow_incomplete,
        ),
        "isolated_sft": _collect_run(
            name="isolated_sft",
            inference_dir=layout.isolated_train_dir / "inference_core_256",
            train_dir=layout.isolated_train_dir,
            expected_image_mode="full",
            benchmark=layout.benchmark,
            benchmark_sha=benchmark_sha,
            benchmark_records=len(benchmark_rows),
            benchmark_ids=set(benchmark_ids),
            composition_program_counts=composition_contract["program_counts"],
            composition_record_ids_by_program=composition_contract["record_ids_by_program"],
            model=layout.model,
            expected_schedule=layout.isolated_schedule,
            expected_schedule_sha=isolated_schedule_sha,
            allow_incomplete=allow_incomplete,
        ),
    }
    _validate_inference_contracts(runs)
    complete_names = [name for name in RUN_ORDER if runs[name]["status"] == "complete"]
    missing_names = [name for name in RUN_ORDER if runs[name]["status"] != "complete"]
    deltas: dict[str, Any] = {}
    for label, left, right in (
        ("episode_minus_isolated", "episode_sft", "isolated_sft"),
        ("episode_minus_base_full", "episode_sft", "base_full"),
        ("isolated_minus_base_full", "isolated_sft", "base_full"),
        ("base_full_minus_no_image", "base_full", "base_no_image"),
    ):
        if left in complete_names and right in complete_names:
            deltas[label] = _metric_deltas(runs[left]["metrics"], runs[right]["metrics"])
    uncertainty = bootstrap_uncertainty(
        runs,
        benchmark_rows,
        set(composition_contract["record_ids"]),
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not missing_names else "incomplete",
        "claim_scope": {
            "label": "single-seed Qwen3-VL-4B effect-size pilot",
            "is_main_result": False,
            "permitted_use": "pipeline validation and effect-size estimation only",
            "prohibited_inference": "not a multi-seed result and not evidence for an 8B main claim",
        },
        "contract": {
            "benchmark_records": len(benchmark_rows),
            "benchmark_sha256": benchmark_sha,
            "heldout_composition": {
                "preregistered_primary_endpoint": True,
                **composition_contract,
            },
            "seed": 17,
            "paired_arms": ["episode_sft", "isolated_sft"],
            "shared_base_model": str(layout.model.resolve()),
        },
        "provenance": {
            "benchmark": _artifact(layout.benchmark),
            "composition_benchmark": _artifact(layout.composition_benchmark),
            "schedule_manifest": _artifact(layout.schedule_manifest),
            "token_profile": _artifact(layout.token_profile),
            "episode_schedule": _artifact(layout.episode_schedule),
            "isolated_schedule": _artifact(layout.isolated_schedule),
            "model": model_inventory(layout.model),
        },
        "runs": runs,
        "complete_runs": complete_names,
        "missing_runs": missing_names,
        "deltas": deltas,
        "sampling_uncertainty": uncertainty,
    }


def _format_percent(value: Any) -> str:
    return "NA" if value is None else f"{float(value) * 100:.2f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a compact companion report without weakening the JSON provenance."""

    lines = [
        "# EpiSpace Qwen3-VL-4B pilot results",
        "",
        "> **Claim scope:** single-seed Qwen3-VL-4B effect-size pilot. This is a pipeline",
        "> validation result, not the multi-seed/8B main result.",
        "",
        f"Status: **{summary['status']}**. Benchmark: "
        f"{summary['contract']['benchmark_records']} records (`{summary['contract']['benchmark_sha256']}`).",
        "",
    ]
    if summary["missing_runs"]:
        lines.extend(["## Missing runs", ""])
        for name in summary["missing_runs"]:
            paths = ", ".join(summary["runs"][name]["missing_artifacts"])
            lines.append(f"- `{name}`: {paths}")
        lines.append("")

    lines.extend(
        [
            "## Aggregate metrics",
            "",
            "Held-out composition is the preregistered primary endpoint: the exact aggregate of "
            "`counterfactual_cross_view.v1` and `target_view_prediction.v1`.",
            "",
            "| Run | Record | Held-out composition | Family exact | Claim pair | Frame pair | Reveal | Triple exact | ACES |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in RUN_ORDER:
        run = summary["runs"][name]
        if run["status"] != "complete":
            lines.append(f"| `{name}` | missing | — | — | — | — | — | — | — |")
            continue
        metrics = run["metrics"]
        cells = [
            _format_percent(metrics[key]["accuracy"])
            for key in (
                "record_accuracy",
                "heldout_composition",
                "family_exact_match",
                "claim_pair_exact_match",
                "frame_pair_exact_match",
                "revealed_accuracy",
                "evidence_triple_exact_match",
            )
        ]
        cells.append(_format_percent(metrics["aces"]["value"]))
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines.append("")

    if summary["deltas"]:
        lines.extend(
            [
                "## Absolute accuracy deltas",
                "",
                "| Contrast (left − right) | Record | Held-out composition | Family | Claim | Frame | Reveal | Triple | ACES |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        delta_keys = AGGREGATE_METRIC_ORDER
        for name, delta in summary["deltas"].items():
            cells = [_format_percent(delta["overall"][key]) for key in delta_keys]
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines.append("")

    uncertainty = summary["sampling_uncertainty"]
    if uncertainty["status"] == "defined":
        lines.extend(
            [
                "## Paired scene-cluster bootstrap",
                "",
                "These 95% intervals quantify sampling uncertainty over benchmark scenes; they do "
                "not replace variance across training seeds.",
                "",
                "| Endpoint | Episode − isolated | 95% percentile CI | Scenes | Records | Resamples |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for key, label in (
            ("record_accuracy", "Record accuracy"),
            ("heldout_composition", "Held-out composition"),
        ):
            endpoint = uncertainty[key]
            lower, upper = endpoint["ci95_percentile"]
            lines.append(
                f"| {label} | {_format_percent(endpoint['observed_delta'])} | "
                f"[{_format_percent(lower)}, {_format_percent(upper)}] | "
                f"{endpoint['scene_clusters']} | {endpoint['records']} | {endpoint['resamples']} |"
            )
        lines.append("")

    complete = summary["complete_runs"]
    if complete:
        programs = sorted(
            {
                program
                for name in complete
                for program in summary["runs"][name]["metrics"]["per_program"]
            }
        )
        lines.extend(
            [
                "## Per-program accuracy",
                "",
                "| Program | " + " | ".join(f"`{name}`" for name in complete) + " |",
                "|---|" + "---:|" * len(complete),
            ]
        )
        for program in programs:
            cells = []
            for name in complete:
                metric = summary["runs"][name]["metrics"]["per_program"].get(program)
                cells.append(_format_percent(metric["accuracy"] if metric else None))
            lines.append(f"| `{program}` | " + " | ".join(cells) + " |")
        lines.append("")

    provenance = summary["provenance"]
    lines.extend(
        [
            "## Provenance anchors",
            "",
            f"- Model inventory: `{provenance['model']['inventory_sha256']}` "
            f"({provenance['model']['file_count']} config/index/weight files)",
            f"- Composition benchmark: `{provenance['composition_benchmark']['sha256']}`",
            f"- Schedule manifest: `{provenance['schedule_manifest']['sha256']}`",
            f"- Token profile: `{provenance['token_profile']['sha256']}`",
            f"- Episode schedule: `{provenance['episode_schedule']['sha256']}`",
            f"- Isolated schedule: `{provenance['isolated_schedule']['sha256']}`",
            "",
            "The JSON companion contains every file-level hash, adapter inventory, evaluator binding, and per-program delta.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary(summary: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

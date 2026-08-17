"""Reusable one-scene, one-capability planning/render/compile loop."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .behavior import RenderSceneView, layout_from_scene_ir
from .collection import render_plan_payload
from .compiler import CapabilityCompiler
from .family import FamilyBlocked, build_question_group
from .generate import generate_plans
from .library import SCRIPT_LIBRARY
from .standards import STD_V1, CompileStandard

SINGLE_RUN_SCHEMA_VERSION = "scriptgen_single_run.v1"
_DETERMINISTIC_RENDER_FAILURE_PREFIXES = (
    "renderer_failure:RuntimeError:auxiliary camera failed physics clearance:",
    "renderer_failure:RuntimeError:scripted path failed traversability clearance:",
    "renderer_failure:RuntimeError:scripted path failed physics capsule clearance:",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_recipe(source_recipe: Path, plan_path: Path, out: Path, seed: int) -> None:
    payload = yaml.safe_load(source_recipe.read_text(encoding="utf-8"))
    payload["recipe_id"] = f"scriptgen_single_seed{seed}_{plan_path.stem}"
    payload["seed"] = seed
    trajectory = payload["trajectory"]
    trajectory["sampling_strategy"] = "scripted_plan"
    trajectory["trajectory_class"] = "T1"
    trajectory["plan_path"] = str(plan_path.resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(out)


def _validate_source(scene_ir: Path, source_recipe: Path) -> None:
    snapshot_path = scene_ir.parent / "scene_snapshot.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"scene snapshot is required next to scene_ir: {snapshot_path}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    recipe = yaml.safe_load(source_recipe.read_text(encoding="utf-8"))
    source = recipe["source"]
    comparisons = {
        "scene_model": (snapshot.get("scene_model"), source.get("scene_model")),
        "scene_instance": (snapshot.get("scene_instance"), source.get("scene_instance")),
        "source_version": (snapshot.get("source_version"), source.get("source_version")),
    }
    mismatches = {
        name: {"scene": scene_value, "recipe": recipe_value}
        for name, (scene_value, recipe_value) in comparisons.items()
        if scene_value != recipe_value
    }
    if mismatches:
        raise ValueError(f"scene_ir source and replay recipe differ: {mismatches}")


def _runtime_env(og_root: Path, data_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    epispace_src = Path(__file__).resolve().parents[2]
    python_path = os.pathsep.join((str(og_root / "src"), str(epispace_src)))
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env.update(
        {
            "PYTHONPATH": python_path,
            "OMNIGIBSON_DATA_PATH": str(data_root),
            "OMNIGIBSON_HEADLESS": "True",
            "OMNI_KIT_ACCEPT_EULA": "YES",
        }
    )
    # A forwarded SSH X11 display can make Kit wait in gpu_foundation even
    # though this acquisition is explicitly headless.
    env.pop("DISPLAY", None)
    return env


def _preflight(og_root: Path, data_root: Path, conda_env: str) -> None:
    if not (og_root / "src" / "omnigibson_episode" / "cli.py").is_file():
        raise FileNotFoundError(f"custom omnigibson_episode backend not found: {og_root}")
    required = (
        data_root / "behavior-1k-assets" / "VERSION",
        data_root / "omnigibson-robot-assets" / "VERSION",
        data_root / "omnigibson.key",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OmniGibson runtime data is incomplete: {missing}")
    conda = shutil.which("conda")
    if conda is None:
        raise FileNotFoundError("conda executable not found")
    check = subprocess.run(
        [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            "-c",
            (
                "import importlib.util;"
                "mods=('isaacsim','omnigibson','omnigibson_episode','numpy');"
                "assert all(importlib.util.find_spec(m) for m in mods)"
            ),
        ],
        cwd=og_root,
        env=_runtime_env(og_root, data_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(f"OmniGibson preflight failed: {check.stdout}{check.stderr}")


def _validate_output_root(output_root: Path, *inputs: Path) -> None:
    protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if output_root in protected:
        raise ValueError(f"refusing to use a broad protected output root: {output_root}")
    contained_inputs = [path for path in inputs if path.is_relative_to(output_root)]
    if contained_inputs:
        raise ValueError(
            f"output root contains required input files and cannot be replaced: {contained_inputs}"
        )


def _render_failure_is_retryable(reason: str | None) -> bool:
    """Whether an unchanged candidate can benefit from one renderer retry."""
    return not (
        reason
        and any(
            reason.startswith(prefix)
            for prefix in _DETERMINISTIC_RENDER_FAILURE_PREFIXES
        )
    )


def _render(
    *,
    recipe: Path,
    bundle: Path,
    log_path: Path,
    og_root: Path,
    data_root: Path,
    conda_env: str,
    gpu_id: int,
    timeout_minutes: int,
) -> tuple[bool, str | None]:
    conda = shutil.which("conda")
    assert conda is not None
    command = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        "-m",
        "omnigibson_episode.cli",
        "acquire",
        "--recipe",
        str(recipe),
        "--output",
        str(bundle),
        "--gpu-id",
        str(gpu_id),
        "--headless",
    ]
    if bundle.exists():
        command.append("--overwrite")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=og_root,
            env=_runtime_env(og_root, data_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            # ``conda run`` launches descendants. A timeout must own a
            # process group or the renderer can survive and race the retry
            # while publishing the same bundle.
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_minutes * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = process.wait()

    report_path = bundle / "render_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "success":
            # The backend atomically publishes the complete directory before
            # OmniGibson shutdown. A shutdown hang / non-zero exit therefore
            # must not discard or re-render an already complete bundle.
            return True, None
        return False, "render_report_not_success"

    failure_path = bundle / "failure_report.json"
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        error = failure.get("error") or {}
        return False, (
            f"renderer_failure:{error.get('type', 'unknown')}:"
            f"{error.get('message', 'unknown')}"
        )
    if timed_out:
        return False, f"render_timeout_after_{timeout_minutes}_minutes"
    if returncode != 0:
        return False, f"renderer_exit_{returncode}"
    return False, "render_report_missing"


def run_one(
    *,
    scene_ir: Path,
    source_recipe: Path,
    output_root: Path,
    capability: str,
    seed: int,
    attempts_per_binding: int,
    candidate_plans: int,
    max_render_candidates: int,
    og_root: Path,
    conda_env: str,
    data_root: Path,
    gpu_id: int,
    timeout_minutes: int = 20,
    overwrite: bool = False,
    std: CompileStandard = STD_V1,
) -> Path:
    """Render candidates sequentially and retain the first authoritative pass."""
    if capability not in SCRIPT_LIBRARY:
        raise ValueError(f"unknown capability: {capability}")
    if candidate_plans <= 0 or max_render_candidates <= 0:
        raise ValueError("candidate counts must be positive")
    scene_ir = scene_ir.resolve()
    source_recipe = source_recipe.resolve()
    output_root = output_root.resolve()
    og_root = og_root.resolve()
    data_root = data_root.resolve()
    _validate_output_root(output_root, scene_ir, source_recipe)
    _validate_source(scene_ir, source_recipe)
    _preflight(og_root, data_root, conda_env)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"single-run output exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    script = SCRIPT_LIBRARY[capability]
    layout = layout_from_scene_ir(scene_ir, std=std)
    generation = generate_plans(
        layout,
        script,
        std,
        seed=seed,
        attempts_per_binding=attempts_per_binding,
        # A one-episode closure should spend its rendering budget across
        # distinct target bindings.  Ten same-target geometry variants are
        # poor fallbacks when that target's conservative OBB occlusion does
        # not survive the real mesh render.
        plans_per_binding=1,
        max_plans=candidate_plans,
    )
    _write_json(output_root / "generation.report.json", generation.model_dump(mode="json"))
    if not generation.plans:
        result_path = output_root / "single_run.result.json"
        _write_json(
            result_path,
            {
                "schema_version": SINGLE_RUN_SCHEMA_VERSION,
                "status": "failed",
                "reason": "geometry_generation_exhausted",
                "capability": capability,
                "standard_version": std.standard_version,
                "seed": seed,
                "attempts": [],
            },
        )
        raise RuntimeError(f"geometry generation exhausted; see {result_path}")

    attempts: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    plans = list(generation.plans)
    if capability == "self_motion_update_occluded":
        object_sizes = {obj.name: obj.size_m for obj in layout.objects}
        obstacle_sizes = {
            obstacle.entity_id: 2.0 * max(obstacle.half_extents_xy)
            for obstacle in layout.occlusion_obstacles
            if obstacle.entity_id is not None
        }

        def occlusion_priority(plan: Any) -> float:
            target_size = object_sizes.get(plan.binding["target"], 0.0)
            witness = plan.clause_witnesses.get("occluded_at_question", {})
            blockers = witness.get("blocked_by", ())
            blocker_size = max(
                (object_sizes.get(name, obstacle_sizes.get(name, 0.0)) for name in blockers),
                default=0.0,
            )
            return blocker_size / max(target_size, 1e-9)

        # OBBs are conservative; a blocker much larger than its target is far
        # more likely to remain a full occlusion in the real instance mask.
        plans.sort(key=occlusion_priority, reverse=True)

    for index, plan in enumerate(plans[:max_render_candidates]):
        job_id = f"candidate-{index:03d}"
        plan_record = output_root / "plans" / f"{job_id}.record.json"
        render_plan = output_root / "plans" / f"{job_id}.views.json"
        recipe = output_root / "recipes" / f"{job_id}.yaml"
        bundle = output_root / "bundles" / job_id
        group = output_root / "groups" / job_id
        compile_path = group / "primary.certificate.json"
        attribution_path = group / "occlusion_attribution.json"
        _write_json(plan_record, plan.model_dump(mode="json"))
        _write_json(render_plan, render_plan_payload(plan, layout, std))
        _write_recipe(source_recipe, render_plan, recipe, seed)
        attempt: dict[str, Any] = {
            "job_id": job_id,
            "plan_id": plan.plan_id,
            "plan_record": str(plan_record),
            "render_plan": str(render_plan),
            "recipe": str(recipe),
            "bundle": str(bundle),
            "status": "rendering",
        }
        attempts.append(attempt)
        ok, reason = _render(
            recipe=recipe,
            bundle=bundle,
            log_path=output_root / "logs" / f"{job_id}.log",
            og_root=og_root,
            data_root=data_root,
            conda_env=conda_env,
            gpu_id=gpu_id,
            timeout_minutes=timeout_minutes,
        )
        if not ok:
            attempt.update(status="failed", reason=reason)
            _write_json(output_root / "single_run.partial.json", attempts)
            continue

        view = RenderSceneView.from_bundle(bundle, std, scene_ir=scene_ir)
        certificate = CapabilityCompiler(script, std).compile(
            view,
            dict(plan.binding),
            geometry_plan=plan.model_dump(mode="json"),
        )
        _write_json(compile_path, certificate.model_dump(mode="json"))
        occlusion_outcome = next(
            (
                row
                for row in certificate.clause_outcomes
                if row.predicate == "occluded_in_view"
            ),
            None,
        )
        if occlusion_outcome is not None:
            _write_json(
                attribution_path,
                {
                    "schema_version": "scriptgen_occlusion_attribution.v1",
                    "plan_id": plan.plan_id,
                    "scene_id": plan.scene_id,
                    "standard_version": std.standard_version,
                    "status": "pass" if occlusion_outcome.holds is True else "fail",
                    "evidence": occlusion_outcome.witness,
                },
            )
        if certificate.status != "answerable" or certificate.mismatch is not None:
            attempt.update(
                status="failed",
                reason=certificate.reason or certificate.mismatch or "compile_not_answerable",
                certificate=str(compile_path),
                occlusion_attribution=(
                    str(attribution_path) if attribution_path.is_file() else None
                ),
            )
            _write_json(output_root / "single_run.partial.json", attempts)
            continue
        try:
            group_path = build_question_group(
                bundle,
                plan_record,
                scene_ir,
                group,
                std,
                seed=seed,
            )
        except (FamilyBlocked, OSError, KeyError, ValueError) as error:
            attempt.update(
                status="failed",
                reason=f"question_group:{type(error).__name__}:{error}",
                certificate=str(compile_path),
            )
            _write_json(output_root / "single_run.partial.json", attempts)
            continue
        attempt.update(
            status="accepted",
            certificate=str(compile_path),
            occlusion_attribution=(
                str(attribution_path) if attribution_path.is_file() else None
            ),
            question_group=str(group_path),
            primary_label=certificate.answer.label if certificate.answer else None,
        )
        accepted = attempt
        break

    result_path = output_root / "single_run.result.json"
    payload = {
        "schema_version": SINGLE_RUN_SCHEMA_VERSION,
        "status": "accepted" if accepted is not None else "failed",
        "reason": None if accepted is not None else "render_candidates_exhausted",
        "capability": capability,
        "standard_version": std.standard_version,
        "seed": seed,
        "accepted": accepted,
        "attempts": attempts,
    }
    _write_json(result_path, payload)
    partial = output_root / "single_run.partial.json"
    if partial.exists():
        partial.unlink()
    if accepted is None:
        raise RuntimeError(f"all render candidates failed; see {result_path}")
    return result_path

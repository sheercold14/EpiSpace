"""Command-line entrypoint for dependency diagnosis, acquisition and compilation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from omnigibson_episode.config import load_recipe


def _doctor() -> tuple[int, dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    data_root = Path(
        os.environ.get("OMNIGIBSON_DATA_PATH", project_root / ".data" / "omnigibson")
    ).expanduser()
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("isaacsim", "omnigibson", "bddl", "torch", "numpy", "yaml")
    }
    gpu = {"available": False, "devices": []}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        gpu = {
            "available": result.returncode == 0,
            "devices": [line.strip() for line in result.stdout.splitlines() if line.strip()],
        }
    library_root = project_root / ".data" / "sysroot" / "usr" / "lib" / "x86_64-linux-gnu"
    requirements = {
        "headless_libraries": {
            "ready": all(
                (library_root / name).exists() for name in ("libGLU.so.1", "libOpenGL.so.0")
            ),
            "path": str(library_root),
        },
        "behavior_assets": {
            "ready": (data_root / "behavior-1k-assets" / "VERSION").is_file(),
            "path": str(data_root / "behavior-1k-assets"),
        },
        "robot_assets": {
            "ready": (data_root / "omnigibson-robot-assets" / "VERSION").is_file(),
            "path": str(data_root / "omnigibson-robot-assets"),
        },
        "behavior_key": {
            "ready": (data_root / "omnigibson.key").is_file(),
            "path": str(data_root / "omnigibson.key"),
        },
    }
    environment_ready = (
        all(modules.values()) and gpu["available"] and requirements["headless_libraries"]["ready"]
    )
    ready = environment_ready and all(
        requirements[name]["ready"] for name in ("behavior_assets", "robot_assets", "behavior_key")
    )
    report = {
        "protocol_version": "omnigibson_episode_doctor.v3",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "modules": modules,
        "gpu": gpu,
        "requirements": requirements,
        "environment_ready": environment_ready,
        "ready": ready,
    }
    return (0 if report["ready"] else 2), report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="og-episode")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--recipe", type=Path, required=True)
    acquire.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--gpu-id", type=int, default=0)
    acquire.add_argument("--headless", action="store_true")
    acquire.add_argument("--overwrite", action="store_true")
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--bundle", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--generator-version", default="omnigibson-spatial-episode/0.1.0")
    compile_parser.add_argument("--maximum-queries-per-relation", type=int, default=2)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--bundle", type=Path, required=True)
    present_parser = commands.add_parser("present")
    present_parser.add_argument("--bundle", type=Path, required=True)
    present_parser.add_argument("--output", type=Path)
    inventory_parser = commands.add_parser("inventory")
    inventory_parser.add_argument("--assets-root", type=Path, required=True)
    inventory_parser.add_argument("--output", type=Path, required=True)
    plan_parser = commands.add_parser("plan-sweep")
    plan_parser.add_argument("--inventory", type=Path, required=True)
    plan_parser.add_argument("--base-recipe", type=Path, required=True)
    plan_parser.add_argument("--output-root", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--seed", type=int)
    plan_parser.add_argument("--scene-overrides", type=Path)
    plan_parser.add_argument("--split-manifest", type=Path)
    refresh_parser = commands.add_parser("refresh-sweep")
    refresh_parser.add_argument("--plan", type=Path, required=True)
    derive_parser = commands.add_parser("refresh-derived")
    derive_parser.add_argument("--plan", type=Path, required=True)
    derive_parser.add_argument("--scene", action="append")
    derive_parser.add_argument("--generator-version", default="omnigibson-spatial-episode/0.1.0")
    run_parser = commands.add_parser("run-sweep")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--gpu-ids", default="0")
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--scene", action="append")
    run_parser.add_argument("--variant", action="append")
    run_parser.add_argument("--retry-failed", action="store_true")
    run_parser.add_argument("--acquire-timeout-minutes", type=int, default=20)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--bundle", type=Path, required=True)
    review_parser = commands.add_parser("review-sweep")
    review_parser.add_argument("--plan", type=Path, required=True)
    review_parser.add_argument("--output", type=Path)
    portal_parser = commands.add_parser("review-portal")
    portal_parser.add_argument("--static-plan", type=Path, required=True)
    portal_parser.add_argument("--intervention-plan", type=Path, action="append")
    portal_parser.add_argument("--output", type=Path, required=True)
    objects_parser = commands.add_parser("object-inventory")
    objects_parser.add_argument("--assets-root", type=Path, required=True)
    objects_parser.add_argument("--output", type=Path, required=True)
    objects_parser.add_argument("--summary-output", type=Path)
    split_parser = commands.add_parser("split-manifest")
    split_parser.add_argument("--inventory", type=Path, required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--validation-count", type=int, default=7)
    split_parser.add_argument("--test-count", type=int, default=7)
    intervention_parser = commands.add_parser("plan-interventions")
    intervention_parser.add_argument("--bundle", type=Path, required=True)
    intervention_parser.add_argument("--object-inventory", type=Path, required=True)
    intervention_parser.add_argument("--output", type=Path)
    intervention_parser.add_argument("--maximum-proposals", type=int, default=20)
    model_swap_parser = commands.add_parser("plan-model-swaps")
    model_swap_parser.add_argument("--bundle", type=Path, required=True)
    model_swap_parser.add_argument("--object-inventory", type=Path, required=True)
    model_swap_parser.add_argument("--output", type=Path)
    model_swap_parser.add_argument("--maximum-proposals", type=int, default=20)
    model_swap_parser.add_argument("--maximum-baseline-overlap-count", type=int, default=1)
    model_swap_parser.add_argument("--minimum-nonoverlap-clearance-m", type=float, default=0.05)
    model_swap_parser.add_argument("--maximum-aspect-ratio-factor", type=float, default=1.5)
    model_swap_parser.add_argument(
        "--execution-mode",
        choices=("physics_settle", "kinematic_counterfactual"),
        default="physics_settle",
    )
    model_swap_parser.add_argument("--settle-steps", type=int, default=12)
    execute_intervention = commands.add_parser("acquire-intervention")
    execute_intervention.add_argument("--recipe", type=Path, required=True)
    execute_intervention.add_argument("--base-bundle", type=Path, required=True)
    execute_intervention.add_argument("--plan", type=Path, required=True)
    execute_intervention.add_argument("--proposal-id", required=True)
    execute_intervention.add_argument("--output", type=Path, required=True)
    execute_intervention.add_argument("--gpu-id", type=int, default=0)
    execute_intervention.add_argument("--headless", action="store_true")
    execute_intervention.add_argument("--overwrite", action="store_true")
    execute_intervention.add_argument("--settle-steps", type=int, default=12)
    pair_parser = commands.add_parser("certify-intervention")
    pair_parser.add_argument("--base-bundle", type=Path, required=True)
    pair_parser.add_argument("--variant-bundle", type=Path, required=True)
    pair_parser.add_argument("--output", type=Path)
    release_parser = commands.add_parser("build-release")
    release_parser.add_argument("--sweep-plan", type=Path, required=True)
    release_parser.add_argument("--split-manifest", type=Path, required=True)
    release_parser.add_argument("--output", type=Path, required=True)
    release_parser.add_argument("--release-id", required=True)
    release_parser.add_argument("--human-review", type=Path)
    release_parser.add_argument("--freeze", action="store_true")
    verify_release = commands.add_parser("verify-release")
    verify_release.add_argument("--manifest", type=Path, required=True)
    verify_release.add_argument("--data-root", type=Path, required=True)
    pair_release = commands.add_parser("build-pair-release")
    pair_release.add_argument("--plan", type=Path, required=True)
    pair_release.add_argument("--output", type=Path, required=True)
    pair_release.add_argument("--release-id", required=True)
    pair_release.add_argument("--human-review", type=Path)
    pair_release.add_argument("--freeze", action="store_true")
    verify_pair_release = commands.add_parser("verify-pair-release")
    verify_pair_release.add_argument("--manifest", type=Path, required=True)
    verify_pair_release.add_argument("--static-data-root", type=Path, required=True)
    verify_pair_release.add_argument("--intervention-data-root", type=Path, required=True)
    intervention_sweep = commands.add_parser("plan-intervention-sweep")
    intervention_sweep.add_argument("--static-sweep-plan", type=Path, required=True)
    intervention_sweep.add_argument("--object-inventory", type=Path, required=True)
    intervention_sweep.add_argument("--split-manifest", type=Path, required=True)
    intervention_sweep.add_argument("--output-root", type=Path, required=True)
    intervention_sweep.add_argument("--output", type=Path, required=True)
    intervention_sweep.add_argument("--sweep-id")
    intervention_sweep.add_argument("--maximum-candidates-per-scene", type=int, default=3)
    intervention_sweep.add_argument(
        "--intervention-kind",
        choices=("relation_flip", "model_swap"),
        default="relation_flip",
    )
    intervention_sweep.add_argument(
        "--model-swap-maximum-baseline-overlap-count", type=int, default=1
    )
    intervention_sweep.add_argument(
        "--model-swap-minimum-nonoverlap-clearance-m", type=float, default=0.05
    )
    intervention_sweep.add_argument(
        "--model-swap-maximum-aspect-ratio-factor", type=float, default=1.5
    )
    intervention_sweep.add_argument(
        "--model-swap-execution-mode",
        choices=("physics_settle", "kinematic_counterfactual"),
        default="physics_settle",
    )
    intervention_sweep.add_argument("--intervention-settle-steps", type=int, default=12)
    refresh_interventions = commands.add_parser("refresh-intervention-sweep")
    refresh_interventions.add_argument("--plan", type=Path, required=True)
    review_interventions = commands.add_parser("review-interventions")
    review_interventions.add_argument("--plan", type=Path, required=True)
    review_interventions.add_argument("--output", type=Path)
    run_interventions = commands.add_parser("run-intervention-sweep")
    run_interventions.add_argument("--plan", type=Path, required=True)
    run_interventions.add_argument("--gpu-ids", default="0")
    run_interventions.add_argument("--workers", type=int, default=1)
    run_interventions.add_argument("--limit", type=int)
    run_interventions.add_argument("--scene", action="append")
    run_interventions.add_argument("--proposal", action="append")
    run_interventions.add_argument("--retry-failed", action="store_true")
    run_interventions.add_argument("--acquire-timeout-minutes", type=int, default=20)
    tasks_parser = commands.add_parser("compile-reasoning-tasks")
    tasks_parser.add_argument("--bundle", type=Path, required=True)
    tasks_parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "doctor":
        code, report = _doctor()
        print(json.dumps(report, indent=2, sort_keys=True))
        return code
    if args.command == "acquire":
        from omnigibson_episode.acquire import acquire_episode

        report = acquire_episode(
            recipe=load_recipe(args.recipe),
            output_directory=args.output,
            gpu_id=args.gpu_id,
            headless=args.headless,
            overwrite=args.overwrite,
        )
        print(report)
        return 0
    if args.command == "inspect":
        from omnigibson_episode.quality import inspect_bundle

        report = inspect_bundle(args.bundle)
        print(args.bundle / "preview.html")
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "present":
        from omnigibson_episode.presentation import build_oral_presentation

        output = build_oral_presentation(args.bundle, args.output)
        print(output)
        return 0
    if args.command == "inventory":
        from omnigibson_episode.scene_inventory import build_scene_inventory

        inventory = build_scene_inventory(args.assets_root, args.output)
        print(args.output)
        print(json.dumps(inventory["summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "plan-sweep":
        from omnigibson_episode.sweep import build_sweep_plan

        plan = build_sweep_plan(
            inventory_path=args.inventory,
            base_recipe_path=args.base_recipe,
            output_root=args.output_root,
            plan_path=args.output,
            seed=args.seed,
            scene_overrides_path=args.scene_overrides,
            split_manifest_path=args.split_manifest,
        )
        print(args.output)
        print(f"jobs={plan['job_count']}")
        return 0
    if args.command == "refresh-sweep":
        from omnigibson_episode.sweep import refresh_sweep_plan

        plan = refresh_sweep_plan(args.plan)
        print(args.plan)
        print(json.dumps(plan["status_counts"], indent=2, sort_keys=True))
        return 0
    if args.command == "refresh-derived":
        from omnigibson_episode.derive import refresh_derived_bundles

        report = refresh_derived_bundles(
            args.plan,
            scene_models=set(args.scene) if args.scene else None,
            generator_version=args.generator_version,
        )
        print(json.dumps(report["counts"], indent=2, sort_keys=True))
        return 0 if report["counts"].get("failed", 0) == 0 else 1
    if args.command == "run-sweep":
        from omnigibson_episode.sweep_runner import run_sweep

        gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
        results = run_sweep(
            plan_path=args.plan,
            gpu_ids=gpu_ids,
            workers=args.workers,
            limit=args.limit,
            scene_models=set(args.scene) if args.scene else None,
            variant_ids=set(args.variant) if args.variant else None,
            retry_failed=args.retry_failed,
            acquisition_timeout_seconds=args.acquire_timeout_minutes * 60,
        )
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0 if all(item["status"] in {"passed", "needs_review"} for item in results) else 1
    if args.command == "audit":
        from omnigibson_episode.reasoning_audit import audit_reasoning_bundle

        report = audit_reasoning_bundle(args.bundle)
        print(args.bundle / "reasoning_audit.json")
        summary = {"status": report["reasoning_status"], **report["evidence_summary"]}
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "review-sweep":
        from omnigibson_episode.review import build_sweep_review

        print(build_sweep_review(args.plan, args.output))
        return 0
    if args.command == "review-portal":
        from omnigibson_episode.review_portal import build_review_portal

        print(
            build_review_portal(
                static_plan=args.static_plan,
                intervention_plans=args.intervention_plan or [],
                output_path=args.output,
            )
        )
        return 0
    if args.command == "object-inventory":
        from omnigibson_episode.object_inventory import build_object_inventory

        inventory = build_object_inventory(args.assets_root, args.output, args.summary_output)
        print(args.output)
        print(json.dumps(inventory["summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "split-manifest":
        from omnigibson_episode.splits import build_split_manifest

        manifest = build_split_manifest(
            args.inventory,
            args.output,
            validation_count=args.validation_count,
            test_count=args.test_count,
        )
        print(args.output)
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
        return 0
    if args.command == "plan-interventions":
        from omnigibson_episode.intervention_plan import plan_relation_flip_interventions

        manifest = plan_relation_flip_interventions(
            bundle_directory=args.bundle,
            object_inventory_path=args.object_inventory,
            output_path=args.output,
            maximum_proposals=args.maximum_proposals,
        )
        print(args.output or args.bundle / "intervention_proposals.json")
        print(f"proposals={manifest['proposal_count']}")
        return 0
    if args.command == "plan-model-swaps":
        from omnigibson_episode.model_swap_plan import plan_model_swap_interventions

        manifest = plan_model_swap_interventions(
            bundle_directory=args.bundle,
            object_inventory_path=args.object_inventory,
            output_path=args.output,
            maximum_proposals=args.maximum_proposals,
            maximum_baseline_overlap_count=args.maximum_baseline_overlap_count,
            minimum_nonoverlap_clearance_m=args.minimum_nonoverlap_clearance_m,
            maximum_aspect_ratio_factor=args.maximum_aspect_ratio_factor,
            execution_mode=args.execution_mode,
            settle_steps=args.settle_steps,
        )
        print(args.output or args.bundle / "model_swap_proposals.json")
        print(f"proposals={manifest['proposal_count']}")
        return 0
    if args.command == "acquire-intervention":
        from omnigibson_episode.intervention_acquire import acquire_intervention

        report = acquire_intervention(
            recipe=load_recipe(args.recipe),
            base_bundle=args.base_bundle,
            intervention_plan_path=args.plan,
            proposal_id=args.proposal_id,
            output_directory=args.output,
            gpu_id=args.gpu_id,
            headless=args.headless,
            overwrite=args.overwrite,
            settle_steps=args.settle_steps,
        )
        print(report)
        return 0
    if args.command == "certify-intervention":
        from omnigibson_episode.minimal_pair import certify_minimal_pair

        certificate = certify_minimal_pair(
            base_bundle=args.base_bundle,
            variant_bundle=args.variant_bundle,
            output_path=args.output,
        )
        print(args.output or args.variant_bundle / "minimal_pair.json")
        print(certificate["pair_id"])
        return 0
    if args.command == "build-release":
        from omnigibson_episode.release import build_release_manifest

        manifest = build_release_manifest(
            sweep_plan_path=args.sweep_plan,
            split_manifest_path=args.split_manifest,
            output_path=args.output,
            release_id=args.release_id,
            human_review_path=args.human_review,
            freeze=args.freeze,
        )
        print(args.output)
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
        return 0
    if args.command == "verify-release":
        from omnigibson_episode.release import verify_release_manifest

        report = verify_release_manifest(manifest_path=args.manifest, data_root=args.data_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    if args.command == "build-pair-release":
        from omnigibson_episode.pair_release import build_pair_release_manifest

        manifest = build_pair_release_manifest(
            plan_path=args.plan,
            output_path=args.output,
            release_id=args.release_id,
            human_review_path=args.human_review,
            freeze=args.freeze,
        )
        print(args.output)
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
        return 0
    if args.command == "verify-pair-release":
        from omnigibson_episode.pair_release import verify_pair_release_manifest

        report = verify_pair_release_manifest(
            manifest_path=args.manifest,
            static_data_root=args.static_data_root,
            intervention_data_root=args.intervention_data_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    if args.command == "plan-intervention-sweep":
        from omnigibson_episode.intervention_sweep import build_intervention_sweep_plan

        plan = build_intervention_sweep_plan(
            static_sweep_plan_path=args.static_sweep_plan,
            object_inventory_path=args.object_inventory,
            split_manifest_path=args.split_manifest,
            output_root=args.output_root,
            plan_path=args.output,
            maximum_candidates_per_scene=args.maximum_candidates_per_scene,
            intervention_kind=args.intervention_kind,
            sweep_id=args.sweep_id,
            model_swap_maximum_baseline_overlap_count=(
                args.model_swap_maximum_baseline_overlap_count
            ),
            model_swap_minimum_nonoverlap_clearance_m=(
                args.model_swap_minimum_nonoverlap_clearance_m
            ),
            model_swap_maximum_aspect_ratio_factor=(args.model_swap_maximum_aspect_ratio_factor),
            model_swap_execution_mode=args.model_swap_execution_mode,
            intervention_settle_steps=args.intervention_settle_steps,
        )
        print(args.output)
        print(f"scenes={plan['scene_count']} jobs={plan['job_count']}")
        return 0
    if args.command == "refresh-intervention-sweep":
        from omnigibson_episode.intervention_sweep import refresh_intervention_sweep_plan

        plan = refresh_intervention_sweep_plan(args.plan)
        print(args.plan)
        print(json.dumps(plan["status_counts"], indent=2, sort_keys=True))
        return 0
    if args.command == "review-interventions":
        from omnigibson_episode.pair_review import build_intervention_review

        print(build_intervention_review(args.plan, args.output))
        return 0
    if args.command == "run-intervention-sweep":
        from omnigibson_episode.intervention_runner import run_intervention_sweep

        results = run_intervention_sweep(
            plan_path=args.plan,
            gpu_ids=[int(value) for value in args.gpu_ids.split(",") if value.strip()],
            workers=args.workers,
            limit=args.limit,
            scene_models=set(args.scene) if args.scene else None,
            proposal_ids=set(args.proposal) if args.proposal else None,
            retry_failed=args.retry_failed,
            acquisition_timeout_seconds=args.acquire_timeout_minutes * 60,
        )
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0 if all(item["status"] in {"certified", "needs_review"} for item in results) else 1
    if args.command == "compile-reasoning-tasks":
        from omnigibson_episode.reasoning_tasks import compile_reasoning_tasks

        manifest = compile_reasoning_tasks(args.bundle, args.output)
        print(args.output or args.bundle / "reasoning_tasks.json")
        print(f"tasks={manifest['task_count']} coverage={manifest['coverage_status']}")
        return 0

    from omnigibson_episode.compile_shared import compile_bundle

    episode = compile_bundle(
        bundle_directory=args.bundle,
        output_path=args.output,
        generator_version=args.generator_version,
        maximum_queries_per_relation=args.maximum_queries_per_relation,
    )
    print(args.output)
    print(f"observations={len(episode.observations)} queries={len(episode.queries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI for planning, packaging and verifying a rendered scriptgen collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collection import (
    DEFAULT_COLLECTION_ID,
    load_collection,
    plan_collection,
    refresh_recipes,
    reject_jobs,
    verify_source_hashes,
)
from .library import SCRIPT_LIBRARY


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build a multi-scene scriptgen review dataset.")
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--scene-root", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    plan.add_argument("--per-capability", type=int, default=2)
    plan.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    verify = commands.add_parser("verify-sources")
    verify.add_argument("--manifest", type=Path, required=True)
    refresh = commands.add_parser("refresh-recipes")
    refresh.add_argument("--manifest", type=Path, required=True)
    render = commands.add_parser("render")
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--og-root", type=Path, required=True)
    render.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    render.add_argument("--workers", type=int)
    render.add_argument("--timeout-minutes", type=int, default=20)
    render.add_argument("--retry-failed", action="store_true")
    render.add_argument("--limit", type=int)
    package = commands.add_parser("package")
    package.add_argument("--manifest", type=Path, required=True)
    replace = commands.add_parser("replace-failed")
    replace.add_argument("--manifest", type=Path, required=True)
    replace.add_argument("--dataset", type=Path, required=True)
    one = commands.add_parser("run-one")
    one.add_argument("--scene-ir", type=Path, required=True)
    one.add_argument("--source-recipe", type=Path, required=True)
    one.add_argument("--output-root", type=Path, required=True)
    one.add_argument("--capability", required=True, choices=sorted(SCRIPT_LIBRARY))
    one.add_argument("--seed", type=int, default=17)
    one.add_argument("--attempts-per-binding", type=int, default=150)
    one.add_argument("--candidate-plans", type=int, default=10)
    one.add_argument("--max-render-candidates", type=int, default=10)
    one.add_argument("--og-root", type=Path, required=True)
    one.add_argument("--conda-env", default="behavior")
    one.add_argument("--data-root", type=Path, required=True)
    one.add_argument("--gpu-id", type=int, default=0)
    one.add_argument("--timeout-minutes", type=int, default=20)
    one.add_argument("--overwrite", action="store_true")
    sources = commands.add_parser("prepare-scenes")
    sources.add_argument("--scene-set", choices=("all",), default="all")
    sources.add_argument("--scenes", nargs="+")
    sources.add_argument("--source-root", type=Path, required=True)
    sources.add_argument("--source-id", default="behavior51_sources_v1")
    sources.add_argument("--og-root", type=Path, required=True)
    sources.add_argument("--conda-env", default="behavior")
    sources.add_argument("--data-root", type=Path, required=True)
    sources.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    sources.add_argument("--workers", type=int)
    sources.add_argument("--accept-eula", action="store_true")
    coverage_plan = commands.add_parser("coverage-plan")
    coverage_plan.add_argument("--source-index", type=Path, required=True)
    coverage_plan.add_argument("--output-root", type=Path, required=True)
    coverage_plan.add_argument("--collection-id", default="behavior51_coverage_v1")
    coverage_plan.add_argument("--accepted-per-binding", type=int, default=10)
    coverage_plan.add_argument("--attempts-per-binding", type=int, default=150)
    coverage_plan.add_argument("--initial-attempts-per-binding", type=int, default=30)
    coverage_plan.add_argument("--maximum-multislot-bindings", type=int, default=128)
    coverage_plan.add_argument("--limit-bindings-per-capability", type=int)
    coverage_plan.add_argument("--capabilities", nargs="+", choices=sorted(SCRIPT_LIBRARY))
    coverage_plan.add_argument("--scenes", nargs="+")
    coverage_plan.add_argument("--initialize-only", action="store_true")
    coverage_run = commands.add_parser("coverage-run")
    coverage_run.add_argument("--manifest", type=Path, required=True)
    coverage_run.add_argument("--og-root", type=Path, required=True)
    coverage_run.add_argument("--conda-env", default="behavior")
    coverage_run.add_argument("--data-root", type=Path, required=True)
    coverage_run.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    coverage_run.add_argument("--workers", type=int)
    coverage_run.add_argument("--timeout-minutes", type=int, default=20)
    coverage_run.add_argument("--limit-cells", type=int)
    coverage_run.add_argument(
        "--cell-ids-file",
        type=Path,
        help="JSON list, or an object containing a cell_ids list, to render",
    )
    coverage_run.add_argument(
        "--no-backfill",
        action="store_true",
        help="render and validate existing candidates without changing the plan manifest",
    )
    coverage_pipeline = commands.add_parser("coverage-pipeline")
    coverage_pipeline.add_argument("--manifest", type=Path, required=True)
    coverage_pipeline.add_argument("--og-root", type=Path, required=True)
    coverage_pipeline.add_argument("--conda-env", default="behavior")
    coverage_pipeline.add_argument("--data-root", type=Path, required=True)
    coverage_pipeline.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    coverage_pipeline.add_argument("--workers", type=int)
    coverage_pipeline.add_argument("--timeout-minutes", type=int, default=20)
    coverage_package = commands.add_parser("coverage-package")
    coverage_package.add_argument("--manifest", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare-scenes":
        if not args.accept_eula:
            raise ValueError("prepare-scenes requires --accept-eula")
        from .source_inventory import prepare_sources

        path = prepare_sources(
            source_root=args.source_root,
            og_root=args.og_root,
            data_root=args.data_root,
            conda_env=args.conda_env,
            gpu_ids=tuple(args.gpu_ids),
            workers=args.workers,
            scenes=tuple(args.scenes) if args.scenes else None,
            source_id=args.source_id,
        )
        print(path)
        return 0
    if args.command == "coverage-plan":
        from .binding_coverage import plan_coverage

        path = plan_coverage(
            source_index_path=args.source_index,
            output_root=args.output_root,
            collection_id=args.collection_id,
            accepted_per_binding=args.accepted_per_binding,
            attempts_per_binding=args.attempts_per_binding,
            initial_attempts_per_binding=args.initial_attempts_per_binding,
            maximum_multislot_bindings=args.maximum_multislot_bindings,
            limit_bindings_per_capability=args.limit_bindings_per_capability,
            capabilities=tuple(args.capabilities) if args.capabilities else None,
            scene_keys=tuple(args.scenes) if args.scenes else None,
            initialize_only=args.initialize_only,
        )
        print(path)
        return 0
    if args.command == "coverage-pipeline":
        from .binding_coverage import run_coverage_pipeline

        path = run_coverage_pipeline(
            args.manifest,
            og_root=args.og_root,
            conda_env=args.conda_env,
            data_root=args.data_root,
            gpu_ids=tuple(args.gpu_ids),
            workers=args.workers,
            timeout_minutes=args.timeout_minutes,
        )
        print(path)
        return 0
    if args.command == "coverage-run":
        from .binding_coverage import run_coverage

        cell_ids = None
        if args.cell_ids_file is not None:
            cell_id_payload = json.loads(args.cell_ids_file.read_text(encoding="utf-8"))
            if isinstance(cell_id_payload, dict):
                cell_id_payload = cell_id_payload.get("cell_ids")
            if not isinstance(cell_id_payload, list) or not all(
                isinstance(cell_id, str) for cell_id in cell_id_payload
            ):
                raise ValueError(
                    "--cell-ids-file must contain a JSON string list or {\"cell_ids\": [...]}"
                )
            cell_ids = tuple(cell_id_payload)
        path = run_coverage(
            args.manifest,
            og_root=args.og_root,
            conda_env=args.conda_env,
            data_root=args.data_root,
            gpu_ids=tuple(args.gpu_ids),
            workers=args.workers,
            timeout_minutes=args.timeout_minutes,
            limit_cells=args.limit_cells,
            cell_ids=cell_ids,
            allow_backfill=not args.no_backfill,
        )
        print(path)
        return 0
    if args.command == "coverage-package":
        from .binding_coverage import package_coverage

        dataset, report = package_coverage(args.manifest)
        print(json.dumps({"dataset": str(dataset), "report": str(report)}, indent=2))
        return 0
    if args.command == "run-one":
        from .single import run_one

        path = run_one(
            scene_ir=args.scene_ir,
            source_recipe=args.source_recipe,
            output_root=args.output_root,
            capability=args.capability,
            seed=args.seed,
            attempts_per_binding=args.attempts_per_binding,
            candidate_plans=args.candidate_plans,
            max_render_candidates=args.max_render_candidates,
            og_root=args.og_root,
            conda_env=args.conda_env,
            data_root=args.data_root,
            gpu_id=args.gpu_id,
            timeout_minutes=args.timeout_minutes,
            overwrite=args.overwrite,
        )
        print(path)
        return 0
    if args.command == "plan":
        path = plan_collection(
            args.scene_root,
            args.output_root,
            collection_id=args.collection_id,
            per_capability=args.per_capability,
            seeds=tuple(args.seeds),
        )
        manifest = load_collection(path)
        print(
            json.dumps(
                {
                    "manifest": str(path),
                    "capabilities": len(SCRIPT_LIBRARY),
                    "jobs": len(manifest.jobs),
                    "failures": len(manifest.failures),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    manifest = load_collection(args.manifest)
    if args.command == "refresh-recipes":
        print(json.dumps({"refreshed_recipes": refresh_recipes(manifest)}))
        return 0
    if args.command == "render":
        from .rendering import render_collection

        path = render_collection(
            manifest,
            og_root=args.og_root.resolve(),
            gpu_ids=tuple(args.gpu_ids),
            workers=args.workers,
            timeout_minutes=args.timeout_minutes,
            retry_failed=args.retry_failed,
            limit=args.limit,
        )
        print(path)
        return 0
    if args.command == "package":
        from .dataset import package_collection

        path = package_collection(manifest)
        print(path)
        return 0
    if args.command == "replace-failed":
        from .dataset import ScriptgenDatasetV1

        dataset = ScriptgenDatasetV1.model_validate_json(args.dataset.read_text(encoding="utf-8"))
        manifest_ids = {job.job_id for job in manifest.jobs}
        dataset_ids = {job.job_id for job in dataset.trajectories}
        if dataset.collection_id != manifest.collection_id or dataset_ids != manifest_ids:
            raise ValueError("dataset does not describe the current collection manifest")
        rejected = {
            job.job_id: job.failure_reason or "unspecified_post_render_failure"
            for job in dataset.trajectories
            if job.status == "failed"
        }
        print(json.dumps({"rejected_jobs": reject_jobs(args.manifest, rejected)}))
        return 0
    verify_source_hashes(manifest)
    print(
        json.dumps({"verified_source_scenes": len({job.scene.scene_key for job in manifest.jobs})})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

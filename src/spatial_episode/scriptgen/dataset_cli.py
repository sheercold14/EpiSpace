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
    return result


def main() -> int:
    args = parser().parse_args()
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

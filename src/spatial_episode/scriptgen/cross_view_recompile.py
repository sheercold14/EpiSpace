"""Recompile old walking cross-view bundles under the current question trio.

Bundles rendered for the retired ``cross_view_pair_relation_k{n}`` spec (and
the old-standard ``cross_view_closer_k{n}`` plans) stay usable as held-out
evaluation data: their trajectories are unchanged, only the gold answers are
recompiled from the render authority under the current ego/anchor/closer
specs.  No rendering happens here.

    .venv/bin/python -m spatial_episode.scriptgen.cross_view_recompile \
        --collection <collection dir> --out <output dir>

The collection directory must contain ``collection.plan.json`` (for the
scene_ir path of every job), ``plans/`` and ``bundles/``.  Each recompiled
trajectory becomes one question group under ``<out>/<job_id>/``; a manifest
records every group and every skip with its reason.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .family import FamilyBlocked, ScriptgenQuestionGroupV1, build_question_group
from .library import CROSS_VIEW_ANCHOR, CROSS_VIEW_CLOSER, CROSS_VIEW_EGO
from .standards import STD_V1

# Old walking-line capabilities whose bundles feed the eval pool.
_OLD_CAPABILITY = re.compile(r"^cross_view_(?:pair_relation|closer)_k([123])$")


def _walking_trio(chain_length: int):
    index = chain_length - 1
    return (CROSS_VIEW_EGO[index], CROSS_VIEW_ANCHOR[index], CROSS_VIEW_CLOSER[index])


def recompile_collection(
    collection: Path, out: Path, *, seed: int = 17
) -> dict[str, object]:
    """Recompile every old cross-view bundle in one collection."""
    collection_plan = json.loads(
        (collection / "collection.plan.json").read_text(encoding="utf-8")
    )
    groups: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for job in collection_plan["jobs"]:
        match = _OLD_CAPABILITY.match(job["capability"])
        if match is None:
            continue
        job_id = job["job_id"]
        bundle = collection / "bundles" / job_id
        plan_record = collection / "plans" / f"{job_id}.record.json"
        scene_ir = Path(job["scene"]["scene_ir"])
        missing = [
            str(path)
            for path in (bundle, plan_record, scene_ir)
            if not path.exists()
        ]
        if missing:
            failures.append({"job_id": job_id, "reason": f"missing:{','.join(missing)}"})
            continue
        try:
            group_path = build_question_group(
                bundle,
                plan_record,
                scene_ir,
                out / job_id,
                STD_V1,
                _walking_trio(int(match.group(1))),
                seed=seed,
                canonical_plan=False,
            )
        except FamilyBlocked as blocked:
            failures.append({"job_id": job_id, "reason": f"blocked:{blocked}"})
            continue
        group = ScriptgenQuestionGroupV1.model_validate_json(
            group_path.read_text(encoding="utf-8")
        )
        groups.append(
            {
                "job_id": job_id,
                "source_capability": job["capability"],
                "group": str(group_path),
                "questions": [
                    {
                        "capability": question.capability,
                        "label": question.label,
                        "skip_reason": question.skip_reason,
                    }
                    for question in group.questions
                ],
            }
        )

    manifest = {
        "schema_version": "cross_view_recompile.v1",
        "source_collection": str(collection),
        "source_standard_version": collection_plan["standard_version"],
        "standard_version": STD_V1.standard_version,
        "pool": "eval_holdout",
        "seed": seed,
        "groups": groups,
        "failures": failures,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "recompile_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Recompile old walking cross-view bundles into eval-pool question groups."
    )
    result.add_argument(
        "--collection", type=Path, required=True, help="Rendered collection directory."
    )
    result.add_argument("--out", type=Path, required=True, help="Output directory.")
    result.add_argument("--seed", type=int, default=17, help="Variant sampling seed.")
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = recompile_collection(args.collection, args.out, seed=args.seed)
    answerable = sum(
        1
        for group in manifest["groups"]
        for question in group["questions"]
        if question["skip_reason"] is None
    )
    print(
        json.dumps(
            {
                "groups": len(manifest["groups"]),
                "failures": len(manifest["failures"]),
                "answerable_questions": answerable,
                "manifest": str(args.out / "recompile_manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

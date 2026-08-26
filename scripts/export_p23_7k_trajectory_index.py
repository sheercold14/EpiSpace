#!/usr/bin/env python
"""Export the audited P2/P3 physical trajectory plan as one flat index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_producer(capability: str) -> bool:
    return capability == "reference_frame_transform" or capability.startswith("cross_view_ego_k")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract, audit = _read(args.contract), _read(args.audit)
    if audit.get("status") != "pass":
        raise RuntimeError("trajectory index requires a passing quality audit")

    trajectories: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for tier, path in (("p2", args.p2), ("p3", args.p3)):
        manifest = _read(path)
        tier_count = 0
        cells = sorted(
            (cell for cell in manifest["cells"] if _is_producer(cell["capability"])),
            key=lambda cell: (cell["scene_key"], cell["capability"], cell["cell_id"]),
        )
        for cell in cells:
            k = int(cell["capability"].rsplit("_k", 1)[1]) if tier == "p3" else None
            for candidate in cell["candidates"]:
                tier_count += 1
                trajectories.append(
                    {
                        "ordinal": len(trajectories),
                        "tier": tier,
                        "k": k,
                        "scene_key": cell["scene_key"],
                        "scene_id": cell["scene_id"],
                        "cell_id": cell["cell_id"],
                        "capability": cell["capability"],
                        "binding": cell["binding"],
                        "candidate_id": candidate["candidate_id"],
                        "plan_id": candidate["plan_id"],
                        "plan_record": candidate["plan_record"],
                        "render_plan": candidate["render_plan"],
                        "recipe": candidate["recipe"],
                    }
                )
        counts[tier] = tier_count

    expected = int(contract["physical_trajectory_target"])
    if len(trajectories) != expected:
        raise RuntimeError(f"trajectory index has {len(trajectories)} rows; needs {expected}")
    payload = {
        "schema_version": "p23_7k_trajectory_index.v1",
        "physical_trajectory_count": len(trajectories),
        "counts": counts,
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "audit": str(args.audit.resolve()),
        "audit_sha256": _sha256(args.audit),
        "manifests": {
            "p2": {"path": str(args.p2.resolve()), "sha256": _sha256(args.p2)},
            "p3": {"path": str(args.p3.resolve()), "sha256": _sha256(args.p3)},
        },
        "trajectories": trajectories,
    }
    _write(args.output, payload)
    print(
        json.dumps(
            {
                "status": "pass",
                "physical_trajectory_count": len(trajectories),
                "counts": counts,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

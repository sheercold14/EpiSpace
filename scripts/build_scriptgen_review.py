"""Export a rendered scripted bundle into a single-page review site.

Collects everything a human needs to judge one planned trajectory — frames,
the script's clauses with their witnesses, the frozen standards, the top-down
layout — into one ``data.json`` plus copied PNGs, next to a self-contained
``index.html``. No server-side logic; open over ``python -m http.server``.

Usage:
    .venv/bin/python scripts/build_scriptgen_review.py \
        --bundle <rendered bundle dir> \
        --plan-record <plan_full_record.json> \
        --scene-ir <source scene_ir.json> \
        --out <output dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.library import SCRIPT_LIBRARY
from spatial_episode.scriptgen.standards import STD_V1

TEMPLATE = Path(__file__).resolve().parent.parent / "web" / "scriptgen_review.html"
CHANNEL_SIZE = 512  # per-channel export resolution


def _save(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).resize((CHANNEL_SIZE, CHANNEL_SIZE), Image.BILINEAR).save(path)


def _depth_to_image(depth_m: np.ndarray) -> np.ndarray:
    """Near = bright, far = dark; robust upper bound at the 99th percentile."""
    finite = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    far = float(np.percentile(finite, 99)) if finite.size else 1.0
    normalized = np.clip(depth_m / max(far, 1e-6), 0.0, 1.0)
    return ((1.0 - normalized) * 255).astype(np.uint8)


def _instance_to_image(instance_id: np.ndarray, target_ids: list[int]) -> np.ndarray:
    """Deterministic color per instance; the target renders bright red."""
    height, width = instance_id.shape
    out = np.zeros((height, width, 3), dtype=np.uint8)
    for uid in np.unique(instance_id):
        mask = instance_id == uid
        if uid in target_ids:
            out[mask] = (230, 40, 40)
        elif uid in (0, 1):  # background / unlabelled
            out[mask] = (28, 28, 34)
        else:
            rng = np.random.default_rng(int(uid))
            out[mask] = rng.integers(70, 220, size=3)
    return out


def _export_channels(
    bundle: Path, target_category: str, frame_count: int, media: Path
) -> list[int]:
    """Write rgb/depth/instance PNGs per frame; return target pixel counts."""
    snapshot = json.loads((bundle / "scene_snapshot.json").read_text(encoding="utf-8"))
    runtime_ids = [
        int(rid)
        for rid, name in snapshot["runtime_instance_registry"].items()
        if target_category.lower() in str(name).lower()
    ]
    pixels: list[int] = []
    for t in range(frame_count):
        with np.load(bundle / "views" / f"view-{t:03d}.sensors.npz") as arrays:
            instance = arrays["instance_id"]
            pixels.append(int(np.isin(instance, runtime_ids).sum()))
            _save(arrays["rgb"], media / f"view-{t:03d}.rgb.png")
            _save(_depth_to_image(arrays["depth_m"]), media / f"view-{t:03d}.depth.png")
            _save(_instance_to_image(instance, runtime_ids), media / f"view-{t:03d}.inst.png")
    return pixels


def build(bundle: Path, plan_record: Path, scene_ir: Path, out: Path) -> Path:
    plan = json.loads(plan_record.read_text(encoding="utf-8"))
    layout = layout_from_scene_ir(scene_ir)
    script = SCRIPT_LIBRARY[plan["capability"]]
    std = STD_V1

    target_id = plan["binding"]["target"]
    target = layout.object(target_id)
    frame_count = len(plan["poses"])
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    pixels = _export_channels(bundle, target.category, frame_count, media)

    t_seen = plan["frame_vars"].get("t_seen")
    t_q = plan["frame_vars"].get("t_q")

    frames = []
    for t in range(frame_count):
        px = pixels[t]
        if px >= std.render_min_visible_pixels:
            tristate = "visible"
        elif px <= std.render_max_invisible_pixels:
            tristate = "invisible"
        else:
            tristate = "ambiguous"
        frames.append(
            {
                "frame": t,
                "rgb": f"media/view-{t:03d}.rgb.png",
                "depth": f"media/view-{t:03d}.depth.png",
                "instance": f"media/view-{t:03d}.inst.png",
                "target_pixels": px,
                "tristate": tristate,
                "is_t_seen": t == t_seen,
                "is_t_q": t == t_q,
            }
        )

    template = script.templates[0]
    question_text = template.text.format(
        target=target.category, **{k: v for k, v in plan["frame_vars"].items()}
    )

    data = {
        "title": plan["plan_id"],
        "scene_id": plan["scene_id"],
        "capability": plan["capability"],
        "standard_version": plan["standard_version"],
        "standards": dataclasses.asdict(std),
        "script": script.model_dump(),
        "plan": plan,
        "target": {"category": target.category, "xy": target.xy, "size_m": target.size_m},
        "layout": {
            "walkable_min": layout.walkable_min,
            "walkable_max": layout.walkable_max,
            "occluders": layout.occluders,
            "objects": [
                {
                    "name": o.name,
                    "category": o.category,
                    "xy": o.xy,
                    "size_m": o.size_m,
                    "is_target": o.name == target_id,
                }
                for o in layout.objects
            ],
        },
        "frames": frames,
        "question": {
            "text": question_text,
            "options": list(template.options),
            "gold": plan["provisional_answer"]["sector"],
            "azimuth_deg": plan["provisional_answer"]["azimuth_deg"],
            "margin_deg": plan["provisional_answer"]["margin_deg"],
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    shutil.copyfile(TEMPLATE, out / "index.html")
    return out / "index.html"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan-record", type=Path, required=True)
    parser.add_argument("--scene-ir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index = build(args.bundle, args.plan_record, args.scene_ir, args.out)
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

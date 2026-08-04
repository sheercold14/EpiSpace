from __future__ import annotations

import json
from pathlib import Path

from episode3d.language import CATEGORY_ZH, entity_name

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "epispace_pilot_v1.json"


def _strict_sweep_categories() -> set[str]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    categories: set[str] = set()
    for source in config["sources"]:
        plan_path = (CONFIG.parent / source["sweep_plan"]).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for job in plan["jobs"]:
            if job.get("status") != "passed":
                continue
            scene_path = Path(job["bundle"]) / "scene_ir.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            categories.update(
                str(entity["raw_label"]).strip().lower().replace(" ", "_")
                for entity in scene["entities"]
            )
    return categories


def test_strict_pilot_sweep_ontology_is_fully_localized() -> None:
    categories = _strict_sweep_categories()

    assert len(categories) == 241
    assert categories <= CATEGORY_ZH.keys()


def test_entity_name_normalizes_source_spacing_and_case() -> None:
    assert entity_name("  Wall Mounted TV ") == "壁挂电视"
    assert entity_name("bottom cabinet no top") == "无台面地柜"


def test_localized_names_preserve_visually_meaningful_distinctions() -> None:
    assert entity_name("fixed_window") == "固定窗"
    assert entity_name("openable_window") == "可开启窗"
    assert entity_name("top_cabinet") == "吊柜"
    assert entity_name("bottom_cabinet") == "地柜"
    assert entity_name("wall_mounted_tv") == "壁挂电视"
    assert entity_name("standing_tv") == "电视"

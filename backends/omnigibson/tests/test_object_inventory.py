import json
from pathlib import Path

from omnigibson_episode.object_inventory import build_object_inventory


def _model(root: Path, category: str, model: str, metadata: dict) -> None:
    target = root / "objects" / category / model
    (target / "misc").mkdir(parents=True)
    (target / "usd").mkdir()
    (target / "misc" / "metadata.json").write_text(json.dumps(metadata))
    (target / "usd" / f"{model}.encrypted.usd").write_bytes(b"usd")


def test_object_inventory_applies_conservative_pair_gate(tmp_path: Path) -> None:
    _model(
        tmp_path,
        "cup",
        "valid",
        {
            "bbox_size": [0.1, 0.1, 0.2],
            "meta_links": {"base_link": {}},
            "link_bounding_boxes": {"base_link": {"collision": {}}},
            "orientations": [],
        },
    )
    _model(
        tmp_path,
        "oversized",
        "invalid",
        {
            "bbox_size": [5.0, 5.0, 5.0],
            "meta_links": {"base_link": {}},
            "link_bounding_boxes": {"base_link": {"collision": {}}},
        },
    )
    output = tmp_path / "catalog.json"
    summary = tmp_path / "summary.json"
    inventory = build_object_inventory(tmp_path, output, summary)
    assert inventory["summary"]["model_count"] == 2
    assert inventory["summary"]["minimal_pair_eligible_model_count"] == 1
    valid = next(model for model in inventory["models"] if model["model"] == "valid")
    assert valid["placement_tier"] == "tabletop"
    assert output.is_file() and summary.is_file()

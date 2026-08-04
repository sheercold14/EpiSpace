import json
from pathlib import Path

from omnigibson_episode.splits import build_split_manifest


def test_split_manifest_is_scene_disjoint_and_exact(tmp_path: Path) -> None:
    scenes = []
    for index in range(20):
        scenes.append(
            {
                "scene_model": f"scene_{index}",
                "domain": "home" if index < 10 else "office",
                "classification": "multi_room_rich",
                "eligibility": {"static_multiview": True},
            }
        )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"scenes": scenes}))
    first = build_split_manifest(inventory, validation_count=3, test_count=4)
    second = build_split_manifest(inventory, validation_count=3, test_count=4)
    assert first == second
    assert first["counts"] == {"test": 4, "train": 13, "validation": 3}
    assert len({item["scene_model"] for item in first["assignments"]}) == 20
    assert first["inventory"] == "inventory.json"
    assert not Path(first["inventory"]).is_absolute()
    assert {item["domain"] for item in first["assignments"] if item["split"] == "test"} == {
        "home",
        "office",
    }

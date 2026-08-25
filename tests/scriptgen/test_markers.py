"""Persistent numeric referent marker contracts."""

from __future__ import annotations

import numpy as np
import pytest

from spatial_episode.scriptgen.markers import (
    annotate_frame,
    assign_badges,
    marker_tokens,
    ordered_binding_items,
)
from spatial_episode.scriptgen.media import export_bundle_channels


def test_p2_and_p3_marker_order_is_semantic_and_stable() -> None:
    p2 = {"target": "t", "facing": "f", "viewpoint": "v"}
    p3 = {"other": "o", "anchor2": "a2", "target": "t", "anchor1": "a1"}

    assert ordered_binding_items(p2) == [
        ("viewpoint", "v"),
        ("facing", "f"),
        ("target", "t"),
    ]
    assert marker_tokens(p2) == {"viewpoint": "①", "facing": "②", "target": "③"}
    assert marker_tokens(p3) == {
        "target": "①",
        "anchor1": "②",
        "anchor2": "③",
        "other": "④",
    }


def test_marker_is_only_drawn_above_the_render_pixel_gate() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    instance = np.zeros((64, 64), dtype=np.int64)
    instance[10:30, 10:30] = 7
    assignments = assign_badges(["chair"], {"chair": (7,)})

    _, accepted = annotate_frame(rgb, instance, assignments, size=64, min_pixels=400)
    _, rejected = annotate_frame(rgb, instance, assignments, size=64, min_pixels=401)

    assert [(item.entity_id, item.number, item.pixels) for item in accepted] == [("chair", 1, 400)]
    assert rejected == []


def test_badged_media_keeps_raw_rgb_and_enforces_two_evidence_frames(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    views = bundle / "views"
    views.mkdir(parents=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    instance = np.zeros((8, 8), dtype=np.int64)
    instance[2:6, 2:6] = 7
    for frame in range(2):
        np.savez(
            views / f"view-{frame:03d}.sensors.npz",
            rgb=rgb,
            depth_m=depth,
            instance_id=instance,
        )
    assignments = assign_badges(["chair"], {"chair": (7,)})

    media = tmp_path / "media"
    export_bundle_channels(
        bundle,
        (7,),
        2,
        media,
        badge_assignments=assignments,
        badge_min_pixels=16,
        badge_min_frames=2,
    )

    assert (media / "view-000.raw.png").is_file()
    assert (media / "view-000.rgb.png").is_file()
    assert (media / "marker.audit.json").is_file()

    with pytest.raises(ValueError, match="marker_visibility_insufficient"):
        export_bundle_channels(
            bundle,
            (7,),
            1,
            tmp_path / "insufficient",
            badge_assignments=assignments,
            badge_min_pixels=16,
            badge_min_frames=2,
        )

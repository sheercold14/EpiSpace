"""Built-in demo layout: a living-room/kitchen scene for render-free runs.

The layout mirrors the kind of scene the OmniGibson backend acquires (sofa,
fridge, TV, a partition wall that occludes) so the whole engine can be
exercised — slotting, motifs, clauses, plans — without Isaac Sim installed.
"""

from __future__ import annotations

from .sceneview import SceneLayout, SceneObject

DEMO_LAYOUT = SceneLayout(
    scene_id="demo_livingroom_v1",
    objects=(
        SceneObject(name="sofa_main", category="sofa", xy=(2.0, 6.5), size_m=1.8, uid="obj_01"),
        SceneObject(name="fridge_main", category="fridge", xy=(8.5, 1.0), size_m=0.9, uid="obj_02"),
        SceneObject(name="tv_main", category="tv", xy=(6.5, 7.5), size_m=1.1, uid="obj_03"),
        SceneObject(name="plant_small", category="plant", xy=(5.0, 4.0), size_m=0.2, uid="obj_04"),
    ),
    # A partition wall between living room and kitchen.
    occluders=(((4.4, 2.5), (4.8, 6.0)),),
    walkable_min=(0.5, 0.5),
    walkable_max=(9.5, 9.5),
)

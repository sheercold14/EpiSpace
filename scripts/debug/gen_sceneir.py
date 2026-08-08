from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.generate import generate_plans
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.standards import STD_V1


SCENE_IR = (
    "/data/shichao/data/dataV100/code/OminiGibson/outputs/sweeps/"
    "t10-target-view-seed17-v1/bundles/"
    "gates_bedroom_t10_seed17/scene_ir.json"
)

layout = layout_from_scene_ir(SCENE_IR)
report = generate_plans(
    layout,
    SELF_MOTION,
    STD_V1,
    seed=17,
    max_plans=1,
)

print(report.plans[0].plan_id, report.rejection_counts)
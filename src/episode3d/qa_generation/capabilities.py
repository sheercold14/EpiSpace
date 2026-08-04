"""SenseNova-SI-aligned capability labels for existing EpiSpace programs."""

from __future__ import annotations

from episode3d.qa_generation.schemas import CapabilitySpec

_CAPABILITIES: dict[str, CapabilitySpec] = {
    "metric_distance": CapabilitySpec(
        "MM", (), "object-object metric distance", "read_metric", "grounded_short"
    ),
    "egocentric_relation": CapabilitySpec(
        "SR", (), "egocentric left/right/front/back", "read_relation", "grounded_short"
    ),
    "cross_view_relation": CapabilitySpec(
        "SR",
        ("PT",),
        "cross-view registration plus canonical relation",
        "read_after_registration",
        "compositional",
    ),
    "counterfactual_verification": CapabilitySpec(
        "CR",
        ("SR", "PT"),
        "multi-view spatial claim verification",
        "verify_composition",
        "compositional",
    ),
    "last_seen_memory": CapabilitySpec(
        "CR", (), "appearance order and extended memory", "read_temporal", "grounded_short"
    ),
    "orbit_identity": CapabilitySpec(
        "PT", (), "view correspondence", "write_correspondence", "grounded_short"
    ),
    "rotation_change_detection": CapabilitySpec(
        "PT", (), "camera rotation reasoning", "write_camera_motion", "grounded_short"
    ),
    "elevation_relation_transfer": CapabilitySpec(
        "PT",
        ("SR",),
        "camera motion and frame-stable relation",
        "write_camera_motion",
        "compositional",
    ),
    "object_centric_perspective": CapabilitySpec(
        "PT", ("SR",), "allocentric object-target transformation", "read_new_frame", "compositional"
    ),
    "target_view_prediction": CapabilitySpec(
        "PT", (), "hypothetical allocentric view", "predict_new_view", "compositional"
    ),
    "grounding_presence": CapabilitySpec(
        "AUX", (), "visual grounding diagnostic", "ground", "atomic_direct"
    ),
    "unknown_abstention": CapabilitySpec(
        "AUX", (), "epistemic abstention diagnostic", "calibrate", "epistemic"
    ),
    "evidence_presence_unknown": CapabilitySpec(
        "AUX", (), "evidence-deletion diagnostic", "calibrate", "epistemic"
    ),
    "evidence_presence_reveal": CapabilitySpec(
        "AUX", (), "evidence-reveal diagnostic", "update_belief", "epistemic"
    ),
    "occlusion_unknown": CapabilitySpec(
        "AUX", (), "occlusion evidence diagnostic", "calibrate", "epistemic"
    ),
    "occlusion_reveal": CapabilitySpec(
        "AUX", (), "occlusion reveal diagnostic", "update_belief", "epistemic"
    ),
}


def capability_for(task_type: str) -> CapabilitySpec:
    try:
        return _CAPABILITIES[task_type]
    except KeyError as error:
        raise ValueError(f"no SenseNova-SI capability mapping for {task_type}") from error


def capability_inventory() -> dict[str, dict[str, object]]:
    return {key: value.as_dict() for key, value in sorted(_CAPABILITIES.items())}


SENSENOVA_CAPABILITIES = ("MM", "SR", "MR", "PT", "CR")

# OmniGibson assets in the current release explicitly lack certified canonical
# object fronts, so T4 orbit identity must remain PT correspondence rather than
# being relabeled as SenseNova MR (visible object side).
CURRENT_UNSUPPORTED_CAPABILITIES = {
    "MR": "current scene_ir declares canonical_front=null and orientation_questions_allowed=false"
}


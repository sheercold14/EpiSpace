"""Script-driven trajectory generation engine.

Layers (see docs/scriptgen_engine.md):

* standards — versioned thresholds, single source of numeric truth;
* predicates — named pure checks with witnesses, shared by search and compile;
* spec/library — declarative per-capability scripts (the only authoring surface);
* slotting/motifs/checker/generate — the generic engine;
* plan — the output contract consumed by acquisition backends.
"""

from .compiler import CapabilityCompiler, Certificate
from .generate import generate_plans
from .library import (
    HOMING,
    MULTI_TURN,
    NET_TURN,
    NET_TURN_MAGNITUDE,
    OCCLUDED_MOTION,
    PURE_ROTATION,
    PURE_TRANSLATION,
    SCRIPT_LIBRARY,
    SELF_MOTION,
    VIEW_SIDE,
)
from .plan import GenerationReport, TrajectoryPlan
from .spec import AnswerSpec, Clause, Knob, ScriptSpec, SlotSpec, Template
from .standards import STD_V1, CompileStandard

__all__ = [
    "HOMING",
    "MULTI_TURN",
    "NET_TURN",
    "NET_TURN_MAGNITUDE",
    "OCCLUDED_MOTION",
    "PURE_ROTATION",
    "PURE_TRANSLATION",
    "SCRIPT_LIBRARY",
    "SELF_MOTION",
    "STD_V1",
    "VIEW_SIDE",
    "AnswerSpec",
    "CapabilityCompiler",
    "Certificate",
    "Clause",
    "CompileStandard",
    "GenerationReport",
    "Knob",
    "ScriptSpec",
    "SlotSpec",
    "Template",
    "TrajectoryPlan",
    "generate_plans",
]

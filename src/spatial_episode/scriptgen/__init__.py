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
from .library import SCRIPT_LIBRARY, SELF_MOTION
from .plan import GenerationReport, TrajectoryPlan
from .spec import Clause, Knob, ScriptSpec, SlotSpec, Template
from .standards import STD_V1, CompileStandard

__all__ = [
    "SCRIPT_LIBRARY",
    "SELF_MOTION",
    "STD_V1",
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

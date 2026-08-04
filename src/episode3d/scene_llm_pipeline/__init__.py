"""Scene-generic dialogue truth pipeline (scales the Rs_int exemplar to sweeps)."""

from .core import SceneDialogueCompiler, ScenePlanError, compile_sweep

__all__ = ["SceneDialogueCompiler", "ScenePlanError", "compile_sweep"]

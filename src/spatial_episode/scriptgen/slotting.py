"""Slot binding: enumerate which scene objects can fill a script's slots."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .sceneview import SceneLayout, SceneObject
from .spec import ScriptSpec, SlotSpec


@dataclass(frozen=True)
class SlotRejection:
    slot: str
    object_name: str
    reason: str


def _qualifies(obj: SceneObject, spec: SlotSpec, layout: SceneLayout) -> str | None:
    """Return a rejection reason, or None if the object fills the slot."""
    if obj.size_m < spec.min_size_m:
        return "too_small"
    if spec.categories and obj.category not in spec.categories:
        return "category_mismatch"
    if spec.unique_referent:
        same_category = [o for o in layout.objects if o.category == obj.category]
        if len(same_category) > 1:
            return "ambiguous_referent"
    return None


def enumerate_bindings(
    layout: SceneLayout, script: ScriptSpec
) -> tuple[list[dict[str, str]], list[SlotRejection]]:
    """All ways to bind scene objects to the script's slots.

    Objects are referenced by name in bindings; a binding never assigns the
    same object to two slots.
    """
    rejections: list[SlotRejection] = []
    candidates: dict[str, list[str]] = {}
    for slot_name, slot_spec in script.slots.items():
        names: list[str] = []
        for obj in layout.objects:
            reason = _qualifies(obj, slot_spec, layout)
            if reason is None:
                names.append(obj.name)
            else:
                rejections.append(SlotRejection(slot_name, obj.name, reason))
        candidates[slot_name] = names

    slot_names = list(script.slots)
    bindings = [
        dict(zip(slot_names, combo, strict=True))
        for combo in product(*(candidates[name] for name in slot_names))
        if len(set(combo)) == len(combo)
    ]
    return bindings, rejections

"""Deterministic project identifiers that do not depend on paths or load order."""

from __future__ import annotations

from uuid import UUID, uuid5

PROJECT_NAMESPACE = UUID("2b74debb-f2c4-5ca6-9563-fbe800f3f724")


def stable_scene_id(source_name: str, source_version: str, source_scene_id: str) -> UUID:
    value = f"scene:{source_name}:{source_version}:{source_scene_id}"
    return uuid5(PROJECT_NAMESPACE, value)


def stable_entity_id(scene_id: UUID, source_entity_id: str, geometry_digest: str) -> UUID:
    value = f"entity:{source_entity_id}:{geometry_digest}"
    return uuid5(scene_id, value)


def stable_region_id(scene_id: UUID, source_region_id: str) -> UUID:
    return uuid5(scene_id, f"region:{source_region_id}")


def stable_episode_family_id(scene_id: UUID, recipe_id: str, family_index: int) -> UUID:
    value = f"family:{recipe_id}:{family_index}"
    return uuid5(scene_id, value)


def stable_episode_id(family_id: UUID, variant_id: str) -> UUID:
    return uuid5(family_id, f"episode:{variant_id}")

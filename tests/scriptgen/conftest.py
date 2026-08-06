"""Shared fixtures for scriptgen tests."""

from __future__ import annotations

import pytest

from spatial_episode.scriptgen.compiler import CapabilityCompiler
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.standards import STD_V1


@pytest.fixture(scope="session")
def self_motion_compiler() -> CapabilityCompiler:
    return CapabilityCompiler(script=SELF_MOTION, std=STD_V1)

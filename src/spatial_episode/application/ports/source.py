"""Port implemented by each source-asset plugin."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from spatial_episode.domain.assets import AssetProbeReport, SourceSceneRef


class SceneSource(Protocol):
    def discover(self) -> Iterable[SourceSceneRef]: ...

    def probe(self, scene: SourceSceneRef) -> AssetProbeReport: ...

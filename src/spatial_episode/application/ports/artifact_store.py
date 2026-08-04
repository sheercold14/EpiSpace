"""Artifact store port."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Protocol

from spatial_episode.contracts.base import ArtifactRefV1


class ArtifactStore(Protocol):
    def put_bytes(self, payload: bytes, *, media_type: str) -> ArtifactRefV1: ...

    def put_file(self, path: Path, *, media_type: str) -> ArtifactRefV1: ...

    def open(self, artifact: ArtifactRefV1) -> BinaryIO: ...

    def exists(self, artifact: ArtifactRefV1) -> bool: ...

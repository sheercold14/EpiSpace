"""Atomic local content-addressed artifact store."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from spatial_episode.contracts.base import ArtifactRefV1


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.blob_root = self.root / "blobs" / "sha256"
        self.blob_root.mkdir(parents=True, exist_ok=True)

    def _path_for_digest(self, digest: str) -> Path:
        return self.blob_root / digest[:2] / digest

    @staticmethod
    def _reference(digest: str, byte_size: int, media_type: str) -> ArtifactRefV1:
        return ArtifactRefV1(
            uri=f"artifact://sha256/{digest}",
            sha256=digest,
            media_type=media_type,
            byte_size=byte_size,
        )

    def put_bytes(self, payload: bytes, *, media_type: str) -> ArtifactRefV1:
        digest = hashlib.sha256(payload).hexdigest()
        destination = self._path_for_digest(digest)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{digest}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return self._reference(digest, len(payload), media_type)

    def put_file(self, path: Path, *, media_type: str) -> ArtifactRefV1:
        return self.put_bytes(path.read_bytes(), media_type=media_type)

    def open(self, artifact: ArtifactRefV1) -> BinaryIO:
        return self._path_for_digest(artifact.sha256).open("rb")

    def exists(self, artifact: ArtifactRefV1) -> bool:
        path = self._path_for_digest(artifact.sha256)
        return path.is_file() and path.stat().st_size == artifact.byte_size

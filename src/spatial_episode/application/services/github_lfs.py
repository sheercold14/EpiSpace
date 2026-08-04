"""Checksum-first GitHub LFS acquisition with direct large-object downloads."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LfsPointer:
    oid_sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class DownloadResult:
    path: Path
    sha256: str
    byte_size: int
    source_kind: str


def parse_lfs_pointer(payload: bytes) -> LfsPointer | None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text.startswith("version https://git-lfs.github.com/spec/v1\n"):
        return None
    fields: dict[str, str] = {}
    for line in text.splitlines()[1:]:
        key, separator, value = line.partition(" ")
        if separator:
            fields[key] = value
    oid = fields.get("oid", "")
    size = fields.get("size", "")
    if not oid.startswith("sha256:") or not size.isdigit():
        raise ValueError("malformed Git LFS pointer")
    digest = oid.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("invalid Git LFS SHA-256")
    return LfsPointer(oid_sha256=digest, byte_size=int(size))


def _no_environment_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request_bytes(request: urllib.request.Request, *, timeout_s: int = 60) -> bytes:
    with _no_environment_proxy_opener().open(request, timeout=timeout_s) as response:
        payload: bytes = response.read()
        return payload


def _download_direct(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    source_kind: str = "git_lfs_direct",
) -> DownloadResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing_size = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "spatial-episode-forge/0.1"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
    request = urllib.request.Request(url, headers=headers)
    opener = _no_environment_proxy_opener()
    with opener.open(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        append = existing_size > 0 and status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)

    byte_size = partial.stat().st_size
    if byte_size != expected_size:
        raise ValueError(f"download size mismatch: got {byte_size}, expected {expected_size}")
    hasher = hashlib.sha256()
    with partial.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"download digest mismatch: got {digest}, expected {expected_sha256}")
    os.replace(partial, destination)
    return DownloadResult(destination, digest, byte_size, source_kind)


def download_github_file(
    *,
    repository: str,
    revision: str,
    repository_path: str,
    destination: Path,
    proxy_base: str = "https://gh-proxy.com/",
    object_proxy_base: str | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> DownloadResult:
    """Download metadata through a GitHub URL proxy and LFS payloads directly.

    Environment proxy variables are deliberately ignored. By default the proxy
    only carries tiny pointer/batch requests and the signed object is direct. On a
    host where GitHub object storage has no direct route, ``object_proxy_base`` may
    name the same GitHub-specific URL mirror. The repository LFS SHA-256 remains
    authoritative in either route.
    """

    proxy = proxy_base.rstrip("/") + "/"
    raw_url = (
        f"https://raw.githubusercontent.com/{repository}/{revision}/{repository_path.lstrip('/')}"
    )
    raw_request = urllib.request.Request(
        proxy + raw_url,
        headers={"User-Agent": "spatial-episode-forge/0.1"},
    )
    raw_payload = _request_bytes(raw_request)
    pointer = parse_lfs_pointer(raw_payload)

    if pointer is None:
        digest = hashlib.sha256(raw_payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"raw file digest mismatch: got {digest}, expected {expected_sha256}")
        if expected_size is not None and len(raw_payload) != expected_size:
            raise ValueError(
                f"raw file size mismatch: got {len(raw_payload)}, expected {expected_size}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.write_bytes(raw_payload)
        os.replace(partial, destination)
        return DownloadResult(destination, digest, len(raw_payload), "git_raw")

    if expected_sha256 is not None and pointer.oid_sha256 != expected_sha256:
        raise ValueError("repository LFS pointer digest differs from locked catalog")
    if expected_size is not None and pointer.byte_size != expected_size:
        raise ValueError("repository LFS pointer size differs from locked catalog")

    batch_url = proxy + f"https://github.com/{repository}.git/info/lfs/objects/batch"
    batch_payload = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": pointer.oid_sha256, "size": pointer.byte_size}],
        }
    ).encode("utf-8")
    batch_request = urllib.request.Request(
        batch_url,
        data=batch_payload,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "spatial-episode-forge/0.1",
        },
        method="POST",
    )
    batch_response = json.loads(_request_bytes(batch_request))
    try:
        signed_url = batch_response["objects"][0]["actions"]["download"]["href"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"Git LFS batch response has no download action: {batch_response}"
        ) from error
    if not isinstance(signed_url, str) or not signed_url.startswith("https://"):
        raise ValueError("Git LFS download action is not an HTTPS URL")
    object_url = signed_url
    source_kind = "git_lfs_direct"
    if object_proxy_base is not None:
        object_url = object_proxy_base.rstrip("/") + "/" + signed_url
        source_kind = "git_lfs_url_mirror"
    return _download_direct(
        object_url,
        destination,
        expected_sha256=pointer.oid_sha256,
        expected_size=pointer.byte_size,
        source_kind=source_kind,
    )

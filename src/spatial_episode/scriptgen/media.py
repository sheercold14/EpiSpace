"""Bundle channel export: sensors.npz -> per-frame review PNGs.

Shared by the single-trajectory review exporter and the family packer
(sunk from scripts/ per the second-use rule). Three aligned channels per
frame: raw RGB, normalised depth, and a deterministic instance colouring
with the target rendered bright red.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

CHANNEL_SIZE = 512  # per-channel export resolution


def save_channel_png(array: np.ndarray, path: Path, size: int = CHANNEL_SIZE) -> None:
    Image.fromarray(array).resize((size, size), Image.BILINEAR).save(path)


def depth_to_image(depth_m: np.ndarray) -> np.ndarray:
    """Near = bright, far = dark; robust upper bound at the 99th percentile."""
    finite = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    far = float(np.percentile(finite, 99)) if finite.size else 1.0
    normalized = np.clip(depth_m / max(far, 1e-6), 0.0, 1.0)
    return ((1.0 - normalized) * 255).astype(np.uint8)


def instance_to_image(instance_id: np.ndarray, target_ids: Sequence[int]) -> np.ndarray:
    """Deterministic color per instance; the target renders bright red."""
    height, width = instance_id.shape
    out = np.zeros((height, width, 3), dtype=np.uint8)
    for uid in np.unique(instance_id):
        mask = instance_id == uid
        if uid in target_ids:
            out[mask] = (230, 40, 40)
        elif uid in (0, 1):  # background / unlabelled
            out[mask] = (28, 28, 34)
        else:
            rng = np.random.default_rng(int(uid))
            out[mask] = rng.integers(70, 220, size=3)
    return out


def export_bundle_channels(
    bundle: Path,
    target_runtime_ids: Sequence[int],
    frame_count: int,
    media_dir: Path,
) -> list[int]:
    """Write rgb/depth/instance PNGs per frame; return target pixel counts."""
    media_dir.mkdir(parents=True, exist_ok=True)
    pixels: list[int] = []
    for t in range(frame_count):
        with np.load(bundle / "views" / f"view-{t:03d}.sensors.npz") as arrays:
            instance = arrays["instance_id"]
            pixels.append(int(np.isin(instance, target_runtime_ids).sum()))
            save_channel_png(arrays["rgb"], media_dir / f"view-{t:03d}.rgb.png")
            save_channel_png(
                depth_to_image(arrays["depth_m"]), media_dir / f"view-{t:03d}.depth.png"
            )
            save_channel_png(
                instance_to_image(instance, target_runtime_ids),
                media_dir / f"view-{t:03d}.inst.png",
            )
    return pixels


def export_named_view_channels(
    view_dir: Path,
    view_ids: Sequence[str],
    target_runtime_ids: Sequence[int],
    media_dir: Path,
) -> list[int]:
    """Export named auxiliary views without treating them as sequence frames."""
    media_dir.mkdir(parents=True, exist_ok=True)
    pixels: list[int] = []
    for view_id in view_ids:
        with np.load(view_dir / f"{view_id}.sensors.npz") as arrays:
            instance = arrays["instance_id"]
            pixels.append(int(np.isin(instance, target_runtime_ids).sum()))
            save_channel_png(arrays["rgb"], media_dir / f"{view_id}.rgb.png")
            save_channel_png(depth_to_image(arrays["depth_m"]), media_dir / f"{view_id}.depth.png")
            save_channel_png(
                instance_to_image(instance, target_runtime_ids),
                media_dir / f"{view_id}.inst.png",
            )
    return pixels

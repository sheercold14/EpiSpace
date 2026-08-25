"""Draw referring badges onto rendered frames.

A question can only ask about an object it can name.  Naming by category alone
needs the category to be unique, and across the fifty-one scenes only 589 of
the 11,273 objects at least 0.3 m across are the only one of their kind in
their scene - 5%.  Another 8,720 sit within two metres of a same-category
twin, close enough that no room name or bearing separates them.

Phrases that count along the picture ("the third table from the left") do not
close the gap either, and for the imagined-viewpoint line they are actively
wrong: the question asks the reader to imagine standing somewhere else, and
"from the left" then names one object from the camera's position and a
different one from the imagined position.  The question would contradict
itself.

A badge painted on the object refers without either problem.  It travels with
the object, so it means the same thing in every frame, and it keeps meaning the
same thing when the imagined viewpoint moves.  It also makes the referent
independent of the geometric visibility model, which cannot predict whether an
object will survive the render: the badge is placed from the rendered instance
mask, so an object is badged exactly when it is actually on screen.

Badges are drawn at a fixed pixel size no matter how far away the object is, so
their size carries no range information beyond what the pixels already carry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .spec import ScriptSpec

# Distinct hues rather than one colour for every badge: the number carries the
# identity, but two badges in one frame are easier to tell apart at a glance
# when they also differ in colour.  The palette is indexed by badge number, not
# by category, so the colour leaks nothing about what the object is.
BADGE_COLORS: tuple[tuple[int, int, int], ...] = (
    (232, 58, 58),
    (46, 134, 222),
    (46, 184, 114),
    (240, 160, 32),
    (156, 92, 214),
    (0, 176, 185),
)
BADGE_TEXT_COLOR = (255, 255, 255)
BADGE_RADIUS_PX = 15  # at the exported resolution, not the render resolution
BADGE_OUTLINE_PX = 2
CONTOUR_WIDTH_PX = 2
BADGE_LEADER_GAP_PX = 4  # clearance between a silhouette and a badge parked outside it

_FONT_CANDIDATES = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")
_CIRCLED_NUMBERS = ("①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨")


@dataclass(frozen=True)
class BadgeAssignment:
    """One badge: which object it names and what number it carries."""

    entity_id: str
    number: int
    runtime_ids: tuple[int, ...]


@dataclass(frozen=True)
class BadgePlacement:
    """Where one badge landed in one frame."""

    number: int
    entity_id: str
    x: int
    y: int
    pixels: int


def script_uses_markers(script: ScriptSpec) -> bool:
    """Whether category placeholders in this script name persistent badges."""
    return script.capability.startswith("reference_frame_") or script.motifs == ("visit_landmarks",)


def ordered_binding_items(binding: Mapping[str, str]) -> list[tuple[str, str]]:
    """Stable semantic badge order: P2 viewpoint/facing/target, P3 chain order."""
    if "viewpoint" in binding:
        preferred = ("viewpoint", "facing", "target")
    else:
        anchors = sorted(
            (slot for slot in binding if slot.startswith("anchor")),
            key=lambda slot: int(slot.removeprefix("anchor")),
        )
        preferred = ("target", *anchors, "other")
    seen = set(preferred)
    trailing = sorted(slot for slot in binding if slot not in seen)
    return [(slot, binding[slot]) for slot in (*preferred, *trailing) if slot in binding]


def marker_tokens(binding: Mapping[str, str]) -> dict[str, str]:
    """Template values matching the numbers assigned by :func:`assign_badges`."""
    items = ordered_binding_items(binding)
    if len(items) > len(_CIRCLED_NUMBERS):
        raise ValueError(f"too many marker referents: {len(items)}")
    return {slot: _CIRCLED_NUMBERS[index] for index, (slot, _) in enumerate(items)}


def assign_badges(
    entity_ids: Sequence[str],
    runtime_by_entity: Mapping[str, tuple[int, ...]],
) -> list[BadgeAssignment]:
    """Number the objects in the order given, skipping any the render cannot show.

    ``runtime_by_entity`` must be the bundle's own map - ``RenderSceneView``
    builds it from ``scene_snapshot.json``.  Runtime instance ids are handed
    out afresh every time a scene is replayed, so the copy in ``scene_ir`` is a
    fallback for bundles that carry no snapshot and matches nothing in a
    replayed episode's masks.

    An entity missing from the map was never handed to the renderer, so it can
    never be badged; numbering it anyway would leave a gap in the question text
    that no frame explains.
    """
    assignments: list[BadgeAssignment] = []
    for entity_id in entity_ids:
        runtime_ids = runtime_by_entity.get(entity_id)
        if not runtime_ids:
            continue
        assignments.append(
            BadgeAssignment(
                entity_id=entity_id,
                number=len(assignments) + 1,
                runtime_ids=runtime_ids,
            )
        )
    return assignments


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def badge_anchor(mask: np.ndarray) -> tuple[int, int, float] | None:
    """The most interior point of the object's largest connected piece, and its clearance.

    Clearance is that point's distance to the piece's boundary - the radius of
    the largest disc the object can hold - which is what decides whether a
    badge fits inside the silhouette or has to be parked outside it.

    The centroid is the obvious choice and the wrong one: a chair seen through
    a table leg splits into two blobs whose centroid falls in the gap, and a
    U-shaped counter's centroid falls outside the counter, so the badge would
    sit on whatever is behind the object it claims to name.  The point furthest
    from the piece's own boundary is always inside it.
    """
    if not mask.any():
        return None
    labels, count = ndimage.label(mask)
    if count > 1:
        sizes = ndimage.sum_labels(mask, labels, index=range(1, count + 1))
        piece = labels == (int(np.argmax(sizes)) + 1)
    else:
        piece = mask
    # Pad so a piece touching the image edge is measured against the edge as a
    # boundary; without the pad the transform treats the edge as interior and
    # anchors the badge half off-screen.
    padded = np.pad(piece, 1, mode="constant", constant_values=False)
    distance = ndimage.distance_transform_edt(padded)
    flat = int(np.argmax(distance))
    y, x = np.unravel_index(flat, distance.shape)
    return int(x) - 1, int(y) - 1, float(distance[y, x])


def _badge_position(mask: np.ndarray, anchor: tuple[int, int, float], size: int) -> tuple[int, int]:
    """Where the badge disc goes: on the object, or parked just above it.

    A badge wider than the silhouette it sits on hides its own referent, which
    leaves the reader a number with nothing under it.  Area is the wrong test
    for that - a long thin television has plenty of area and no room - so the
    test is whether the object can hold a disc the size of the badge.  Those
    that cannot get the badge parked above them, joined by a connector.

    This reveals nothing the picture does not already show: the contour traces
    the silhouette exactly either way, so the placement rule adds no size or
    range cue that was not already on screen.
    """
    x, y, clearance = anchor
    margin = BADGE_RADIUS_PX + BADGE_OUTLINE_PX
    clamp = lambda value: min(max(value, margin), size - margin - 1)  # noqa: E731
    if clearance >= margin + BADGE_LEADER_GAP_PX:
        return clamp(x), clamp(y)
    top = int(np.nonzero(mask.any(axis=1))[0][0])
    return clamp(x), clamp(top - BADGE_RADIUS_PX - BADGE_LEADER_GAP_PX)


def _draw_contour(draw: ImageDraw.ImageDraw, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    """Outline the object's silhouette so a sliver of it is still locatable."""
    border = mask & ~ndimage.binary_erosion(mask, iterations=CONTOUR_WIDTH_PX)
    ys, xs = np.nonzero(border)
    for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
        draw.point((x, y), fill=color)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour, so resizing never grows the object past its own silhouette.

    The mask is resized rather than the instance array it came from: runtime
    instance ids run past 2^31 and PIL's integer image mode is signed 32-bit,
    so putting the id array through a resize wraps the large ids negative and
    every lookup afterwards misses.
    """
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)
    return np.asarray(resized) > 127


def annotate_frame(
    rgb: np.ndarray,
    instance_id: np.ndarray,
    assignments: Sequence[BadgeAssignment],
    *,
    size: int,
    min_pixels: int,
    draw_contour: bool = True,
) -> tuple[np.ndarray, list[BadgePlacement]]:
    """Badge every assigned object that is genuinely on screen in this frame.

    ``min_pixels`` is counted at the render resolution, before the export
    resize, so the same threshold means the same thing whatever the review
    images are scaled to.  An object below it is left unbadged rather than
    badged faintly: a badge on an object the reader cannot make out is a
    question with no evidence behind it.
    """
    image = Image.fromarray(rgb).resize((size, size), Image.BILINEAR).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(int(BADGE_RADIUS_PX * 1.3))

    placements: list[BadgePlacement] = []
    for assignment in assignments:
        full = np.isin(instance_id, assignment.runtime_ids)
        pixels = int(full.sum())
        if pixels < min_pixels:
            continue
        mask = _resize_mask(full, size)
        anchor = badge_anchor(mask)
        if anchor is None:
            continue
        color = BADGE_COLORS[(assignment.number - 1) % len(BADGE_COLORS)]
        if draw_contour:
            _draw_contour(draw, mask, color)
        x, y = _badge_position(mask, anchor, size)
        if (x, y) != anchor[:2]:
            # Parked outside the silhouette: without a connector the badge is
            # a number floating next to two or three candidate objects.
            draw.line((anchor[0], anchor[1], x, y), fill=color, width=CONTOUR_WIDTH_PX)
        draw.ellipse(
            (x - BADGE_RADIUS_PX, y - BADGE_RADIUS_PX, x + BADGE_RADIUS_PX, y + BADGE_RADIUS_PX),
            fill=color,
            outline=BADGE_TEXT_COLOR,
            width=BADGE_OUTLINE_PX,
        )
        draw.text((x, y), str(assignment.number), fill=BADGE_TEXT_COLOR, font=font, anchor="mm")
        placements.append(
            BadgePlacement(
                number=assignment.number,
                entity_id=assignment.entity_id,
                x=x,
                y=y,
                pixels=pixels,
            )
        )
    return np.asarray(image), placements

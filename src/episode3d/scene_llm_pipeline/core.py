"""Scene-generic truth compiler and scenario planner for dialogue episodes.

Design contract (inherits the Rs_int exemplar's four-layer split):
- this module is deterministic; no language model is invoked here;
- every answer sentence is compiled from executed geometry and carries claims;
- surfaces never expose raw axes, entity ids, or oracle channels; directions
  are anchored to the perceivable first-view heading;
- a scene that cannot certify a round skips it; a scene without at least one
  integration round (cross-view or perspective) is rejected with reasons.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .lexicon import CATEGORY_NAMES_ZH, QUADRANT_ZH, RELATION_ZH, STRUCTURAL_LABELS

SCHEMA_VERSION = "epispace.scene_dialogue_truth.v1"
FRAME_ID = "first_view_heading_xy"
FRAME_SURFACE_ZH = "方向问题一律以第1个视角出发时面对的方向为「前」、右手边为「右」来回答（俯视想象这个空间即可）。"
MIN_RELATION_MARGIN_M = 0.4
MIN_PIXELS = 256
HARD_QUADRANT_MIN_ANGLE_DEG = 20.0
RELEASE_SLOTS = ((0,), (1, 2), (3, 4), (5, 6, 7), (8, 9), (10,), ())
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}")
FORBIDDEN_SURFACE_TOKENS = ("+X", "+Y", "+x", "+y", "entity_id", "scene_ir", "relation_oracle", "坐标系", "OBB")


class ScenePlanError(RuntimeError):
    """The bundle cannot support a certified dialogue."""


@dataclass
class EntityFact:
    label: str
    name: str
    entity_id: str
    center_anchor: np.ndarray  # (x_right, y_forward, z_up) in first-view frame
    half_extents: np.ndarray
    rotation_anchor: np.ndarray  # 3x3, entity->anchor frame
    views: tuple[str, ...]


@dataclass
class Round:
    round_index: int
    turn_id: str
    capability: dict[str, Any]
    signature: str
    new_view_ids: list[str]
    evidence_view_ids: list[str]
    nodes: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    question_zh: str
    answer_sentences: list[dict[str, Any]]
    answer_key: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def answer_zh(self) -> str:
        return "".join(s["text"] for s in self.answer_sentences)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScenePlanError(f"expected JSON object: {path}")
    return value


def _quat_matrix(xyzw: list[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _yaw_deg(xyzw: list[float]) -> float:
    x, y, z, w = (float(v) for v in xyzw)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class SceneCatalog:
    """Loads one acquisition bundle and exposes anchor-frame geometry."""

    def __init__(self, bundle_root: Path) -> None:
        self.root = Path(bundle_root)
        self.scene = _read_json(self.root / "scene_ir.json")
        self.episode = _read_json(self.root / "spatial_episode.json")
        trajectory = _read_json(self.root / "trajectory_plan.json")
        quality = _read_json(self.root / "quality_report.json")
        if quality["integrity_status"] != "pass":
            raise ScenePlanError("bundle failed integrity inspection")
        self.family_id = self.episode["family_id"]
        self.split_group = self.episode["split_group"]
        self.view_ids: list[str] = [o["view_id"] for o in self.episode["observations"]]
        if len(self.view_ids) != 11:
            raise ScenePlanError(f"expected 11 views, got {len(self.view_ids)}")
        self.observations = {o["view_id"]: o for o in self.episode["observations"]}
        self.roles = {v["view_id"]: v["role"] for v in trajectory["views"]}
        self.visible: dict[str, set[str]] = {
            o["view_id"]: set(o["visible_entity_ids"]) for o in self.episode["observations"]
        }

        cam0 = self.observations[self.view_ids[0]]["world_from_camera"]
        yaw = math.radians(_yaw_deg(cam0["rotation_xyzw"]))
        forward = np.array([-math.sin(yaw), math.cos(yaw)])
        right = np.array([forward[1], -forward[0]])
        origin = np.array(cam0["translation_m"][:2], dtype=float)
        self._anchor = {"origin": origin, "right": right, "forward": forward}

        runtime_map = self.scene.get("runtime_semantic_id_map", {})
        self._runtime_ids: dict[str, list[int]] = {}
        for runtime_id, entity_id in runtime_map.items():
            self._runtime_ids.setdefault(entity_id, []).append(int(runtime_id))
        self._mask_cache: dict[str, np.ndarray] = {}

        observed_ids = set().union(*self.visible.values())
        by_label: dict[str, list[dict[str, Any]]] = {}
        for raw in self.scene["entities"]:
            label = raw["raw_label"]
            if label in STRUCTURAL_LABELS or raw["entity_id"] not in observed_ids:
                continue
            by_label.setdefault(label, []).append(raw)
        self.entities: dict[str, EntityFact] = {}
        self.unnamed_labels: list[str] = []
        for label, group in sorted(by_label.items()):
            if len(group) != 1:
                continue  # referential ambiguity among observed instances
            if label not in CATEGORY_NAMES_ZH:
                self.unnamed_labels.append(label)
                continue
            raw = group[0]
            world = np.array(raw["world_from_entity"]["translation_m"], dtype=float)
            rel = world[:2] - origin
            center = np.array([float(rel @ right), float(rel @ forward), world[2]])
            rot_world = _quat_matrix(raw["world_from_entity"]["rotation_xyzw"])
            to_anchor = np.array([[right[0], right[1], 0.0], [forward[0], forward[1], 0.0], [0.0, 0.0, 1.0]])
            views = tuple(v for v in self.view_ids if raw["entity_id"] in self.visible[v])
            self.entities[label] = EntityFact(
                label=label,
                name=CATEGORY_NAMES_ZH[label],
                entity_id=raw["entity_id"],
                center_anchor=center,
                half_extents=np.array(raw["obb"]["half_extents_m"], dtype=float),
                rotation_anchor=to_anchor @ rot_world,
                views=views,
            )
        self.unobserved_named: list[str] = sorted(
            label
            for label, group in _group_labels(self.scene["entities"]).items()
            if label in CATEGORY_NAMES_ZH
            and label not in STRUCTURAL_LABELS
            and not any(e["entity_id"] in observed_ids for e in group)
        )

    # -- perception gates -------------------------------------------------
    def visible_pixels(self, label: str, view_id: str) -> int:
        entity = self.entities[label]
        runtime_ids = self._runtime_ids.get(entity.entity_id, [])
        if not runtime_ids:
            return 0
        if view_id not in self._mask_cache:
            with np.load(self.root / "views" / f"{view_id}.sensors.npz") as arrays:
                self._mask_cache[view_id] = arrays["instance_id"].copy()
        return int(np.isin(self._mask_cache[view_id], runtime_ids).sum())

    def well_visible(self, label: str, view_ids: list[str] | tuple[str, ...]) -> str | None:
        """Return the first view where the entity passes the pixel gate."""
        for view_id in view_ids:
            if self.visible_pixels(label, view_id) >= MIN_PIXELS:
                return view_id
        return None

    # -- geometry in the first-view anchor frame --------------------------
    def relation(self, subject: str, reference: str) -> dict[str, Any]:
        ps, pr = self.entities[subject].center_anchor, self.entities[reference].center_anchor
        dx, dy = float(ps[0] - pr[0]), float(ps[1] - pr[1])
        if abs(dx) >= abs(dy):
            rel, dominant, margin = ("right_of" if dx > 0 else "left_of"), abs(dx), abs(dx) - abs(dy)
        else:
            rel, dominant, margin = ("in_front_of" if dy > 0 else "behind"), abs(dy), abs(dy) - abs(dx)
        return {
            "frame": FRAME_ID,
            "subject": subject,
            "reference": reference,
            "relation": rel,
            "dominant_delta_m": round(dominant, 3),
            "margin_m": round(margin, 3),
        }

    def distance(self, a: str, b: str) -> float:
        pa, pb = self.entities[a].center_anchor, self.entities[b].center_anchor
        return float(np.linalg.norm(pa - pb))

    def _obb_half_extent_on(self, label: str, axis_xy: np.ndarray) -> float:
        entity = self.entities[label]
        axis3 = np.array([axis_xy[0], axis_xy[1], 0.0])
        return float(sum(abs(axis3 @ (entity.rotation_anchor[:, i] * entity.half_extents[i])) for i in range(3)))

    def object_anchored(self, origin: str, facing: str, target: str) -> dict[str, Any]:
        po = self.entities[origin].center_anchor[:2]
        pf = self.entities[facing].center_anchor[:2]
        pt = self.entities[target].center_anchor[:2]
        heading = pf - po
        norm = float(np.linalg.norm(heading))
        if norm < 1e-6:
            raise ScenePlanError("coincident anchors for query frame")
        fwd = heading / norm
        rgt = np.array([fwd[1], -fwd[0]])
        q = pt - po
        qx, qy = float(q @ rgt), float(q @ fwd)
        ex = self._obb_half_extent_on(target, rgt)
        ey = self._obb_half_extent_on(target, fwd)
        angle_margin = math.degrees(min(math.atan2(abs(qx), abs(qy)), math.atan2(abs(qy), abs(qx))))
        quadrant = f"{'front' if qy >= 0 else 'back'}-{'right' if qx >= 0 else 'left'}"
        hard_ok = (
            angle_margin >= HARD_QUADRANT_MIN_ANGLE_DEG
            and abs(qx) - ex > 0.0
            and abs(qy) - ey > 0.0
        )
        return {
            "origin": origin,
            "facing": facing,
            "target": target,
            "query_xy_m": [round(qx, 3), round(qy, 3)],
            "obb_half_extents_on_axes_m": [round(ex, 3), round(ey, 3)],
            "angle_margin_deg": round(angle_margin, 2),
            "quadrant": quadrant,
            "hard_quadrant_ok": hard_ok,
        }

    def turnaround(self) -> dict[str, Any] | None:
        for view_id, role in self.roles.items():
            if role == "turnaround":
                index = self.view_ids.index(view_id)
                if index + 1 < len(self.view_ids):
                    a = self.observations[self.view_ids[index - 1]]["world_from_camera"]
                    b = self.observations[self.view_ids[index + 1]]["world_from_camera"]
                    delta = abs(_yaw_deg(b["rotation_xyzw"]) - _yaw_deg(a["rotation_xyzw"])) % 360.0
                    delta = min(delta, 360.0 - delta)
                    return {"view_id": view_id, "view_index": index, "yaw_delta_deg": round(delta, 1)}
        return None

    def loop_closed(self) -> bool:
        a = self.observations[self.view_ids[0]]["world_from_camera"]
        b = self.observations[self.view_ids[-1]]["world_from_camera"]
        return bool(
            np.allclose(a["translation_m"], b["translation_m"], atol=1e-6)
            and np.allclose(a["rotation_xyzw"], b["rotation_xyzw"], atol=1e-6)
        )


def _group_labels(entities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        groups.setdefault(entity["raw_label"], []).append(entity)
    return groups


class SceneDialogueCompiler:
    """Plans and compiles one certified dialogue episode for one bundle."""

    def __init__(self, bundle_root: Path, direction_balance: dict[str, int] | None = None) -> None:
        self.catalog = SceneCatalog(Path(bundle_root))
        self.balance = direction_balance if direction_balance is not None else {}
        self._claim_seq = 0

    # -- helpers -----------------------------------------------------------
    def _viewnum(self, view_id: str) -> int:
        return self.catalog.view_ids.index(view_id) + 1

    def _viewref(self, view_id: str) -> str:
        return f"第{self._viewnum(view_id)}个视角"

    def _claim(self, node_id: str, kind: str, statement: str, value: Any, evidence: list[str], role: str | None = None, required: bool = True) -> dict[str, Any]:
        self._claim_seq += 1
        claim = {
            "claim_id": f"claim-{self._claim_seq:03d}",
            "node_id": node_id,
            "kind": kind,
            "statement_zh": statement,
            "value": value,
            "required": required,
            "evidence_view_ids": evidence,
        }
        if role:
            claim["reasoning_role"] = role
        return claim

    @staticmethod
    def _node(node_id: str, op: str, inputs: list[str], output_type: str, value: Any) -> dict[str, Any]:
        return {"node_id": node_id, "operation": op, "inputs": inputs, "output_type": output_type, "value": value}

    def _pick_balanced(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        candidates.sort(key=lambda c: (self.balance.get(c["relation"], 0), -c["margin_m"]))
        chosen = candidates[0]
        self.balance[chosen["relation"]] = self.balance.get(chosen["relation"], 0) + 1
        return chosen

    # -- round builders (return None when the scene cannot certify them) ---
    def _round_inventory(self, prefix: list[str]) -> Round | None:
        v0 = prefix[0]
        names = [
            self.catalog.entities[label].name
            for label in sorted(self.catalog.entities)
            if v0 in self.catalog.entities[label].views and self.catalog.visible_pixels(label, v0) >= MIN_PIXELS
        ]
        if len(names) < 2:
            return None
        listed = "、".join(names[:8])
        return Round(
            round_index=1,
            turn_id="r01-inventory",
            capability={"primary": "SR", "sense_nova_subtask": "visual grounding"},
            signature="G(view,*)^n->V_presence_set",
            new_view_ids=[v0],
            evidence_view_ids=[v0],
            nodes=[self._node("ground_all", "G", ["view-1"], "entity_set", names[:8]), self._node("verify", "V", ["ground_all"], "boolean", True)],
            claims=[self._claim("ground_all", "grounding", f"第1个视角中可辨认的主要物体：{listed}。", names[:8], [v0], role="conclusion")],
            question_zh="请观察第1个视角的图像，说说里面能看到哪些主要的家具或设备？",
            answer_sentences=[{"role": "conclusion", "text": f"我能看到：{listed}。"}],
            answer_key={"visible_set": names[:8]},
        )

    def _round_covis_relation(self, prefix: list[str], new_views: list[str]) -> Round | None:
        labels = list(self.catalog.entities)
        candidates = []
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                shared = [v for v in prefix if v in self.catalog.entities[a].views and v in self.catalog.entities[b].views]
                shared = [v for v in shared if self.catalog.visible_pixels(a, v) >= MIN_PIXELS and self.catalog.visible_pixels(b, v) >= MIN_PIXELS]
                if not shared:
                    continue
                fact = self.catalog.relation(a, b)
                if fact["margin_m"] < MIN_RELATION_MARGIN_M:
                    continue
                fact["view_id"] = shared[0]
                candidates.append(fact)
        fact = self._pick_balanced(candidates)
        if fact is None:
            return None
        a, b = fact["subject"], fact["reference"]
        na, nb = self.catalog.entities[a].name, self.catalog.entities[b].name
        vref = self._viewref(fact["view_id"])
        distance = self.catalog.distance(a, b)
        return Round(
            round_index=2,
            turn_id="r02-covis-relation",
            capability={"primary": "SR", "supporting": ["MM"], "sense_nova_subtask": "co-visible relation + metric"},
            signature="G+G->B->R+M->V",
            new_view_ids=new_views,
            evidence_view_ids=[fact["view_id"]],
            nodes=[
                self._node("ground_pair", "G", [fact["view_id"], a, b], "entity_pair", [a, b]),
                self._node("relation", "R", ["ground_pair"], "relation", fact["relation"]),
                self._node("metric", "M", ["ground_pair"], "distance_m", round(distance, 3)),
                self._node("verify", "V", ["relation", "metric"], "boolean", fact["margin_m"] >= MIN_RELATION_MARGIN_M),
            ],
            claims=[
                self._claim("ground_pair", "grounding", f"{na}和{nb}在{vref}同框可见。", [a, b], [fact["view_id"]], role="cue"),
                self._claim("relation", "relation", f"以出发朝向为参照，{na}在{nb}的{RELATION_ZH[fact['relation']]}。", fact, [fact["view_id"]], role="conclusion"),
                self._claim("metric", "distance", f"两者中心相距约{round(distance, 1)}米。", {"distance_m": round(distance, 3), "definition": "center_to_center"}, [fact["view_id"]], role="conclusion", required=False),
            ],
            question_zh=f"综合目前看到的画面：有没有哪个视角能同时看到{na}和{nb}？如果有，{na}在{nb}的哪一侧？两者中心大约相距多少米？",
            answer_sentences=[
                {"role": "evidence", "text": f"有，{vref}里两者同时可见。"},
                {"role": "conclusion", "text": f"以出发朝向为参照，{na}在{nb}的{RELATION_ZH[fact['relation']]}，中心相距约{round(distance, 1)}米。"},
            ],
            answer_key={"relation": fact["relation"], "distance_m": round(distance, 3)},
            diagnostics={"margin_m": fact["margin_m"]},
        )

    def _round_temporal(self, prefix: list[str], new_views: list[str]) -> Round | None:
        for label in sorted(self.catalog.entities):
            entity = self.catalog.entities[label]
            seen_prefix = [v for v in prefix if v in entity.views]
            if not seen_prefix:
                continue
            last_seen = seen_prefix[-1]
            if last_seen in new_views:
                continue
            if any(v in entity.views for v in new_views):
                continue
            if self.catalog.visible_pixels(label, last_seen) < MIN_PIXELS:
                continue
            name = entity.name
            refs = "、".join(self._viewref(v) for v in new_views)
            return Round(
                round_index=3,
                turn_id="r03-temporal-recall",
                capability={"primary": "CR", "sense_nova_subtask": "episodic last-seen recall"},
                signature="G(sequence)->B_temporal->V_last_seen",
                new_view_ids=new_views,
                evidence_view_ids=[last_seen, *new_views],
                nodes=[
                    self._node("track", "G", ["sequence", label], "visibility_timeline", [self._viewnum(v) for v in seen_prefix]),
                    self._node("belief", "B", ["track"], "temporal_belief", {"last_seen": self._viewnum(last_seen)}),
                    self._node("verify", "V", ["belief"], "boolean", True),
                ],
                claims=[
                    self._claim("track", "visibility", f"{refs}中都没有{name}。", False, list(new_views), role="cue"),
                    self._claim("belief", "last_seen", f"{name}最后一次出现在{self._viewref(last_seen)}。", self._viewnum(last_seen), [last_seen], role="conclusion"),
                ],
                question_zh=f"{refs}里还能看到{name}吗？它最后一次出现在第几个视角？",
                answer_sentences=[
                    {"role": "conclusion", "text": f"看不到了；{name}最后一次出现是在{self._viewref(last_seen)}。"}
                ],
                answer_key={"visible_now": False, "last_seen_view": self._viewnum(last_seen)},
            )
        return None

    def _round_crossview(self, prefix: list[str], new_views: list[str]) -> Round | None:
        labels = list(self.catalog.entities)
        turnaround = self.catalog.turnaround()
        candidates = []
        for s in labels:
            for r in labels:
                if s == r:
                    continue
                sv = [v for v in prefix if v in self.catalog.entities[s].views]
                rv = [v for v in prefix if v in self.catalog.entities[r].views]
                if not sv or not rv:
                    continue
                if set(self.catalog.entities[s].views) & set(self.catalog.entities[r].views):
                    continue
                if self.catalog.well_visible(s, sv) is None or self.catalog.well_visible(r, rv) is None:
                    continue
                fact = self.catalog.relation(s, r)
                if fact["margin_m"] < MIN_RELATION_MARGIN_M:
                    continue
                fact["subject_views"] = sv
                fact["reference_views"] = rv
                candidates.append(fact)
        fact = self._pick_balanced(candidates)
        if fact is None:
            return None
        s, r = fact["subject"], fact["reference"]
        ns, nr = self.catalog.entities[s].name, self.catalog.entities[r].name
        s_refs = "、".join(self._viewref(v) for v in fact["subject_views"][:3])
        r_refs = "、".join(self._viewref(v) for v in fact["reference_views"][:3])
        sentences = [
            {"role": "evidence", "text": f"两者从未同框：{nr}只出现在{r_refs}，{ns}只出现在{s_refs}。"},
        ]
        claims = [
            self._claim("covis", "co_visibility", f"{ns}与{nr}没有共同可见的视角。", [], fact["subject_views"][:3] + fact["reference_views"][:3], role="cue"),
        ]
        if turnaround:
            sentences.append({"role": "transform", "text": f"{self._viewref(turnaround['view_id'])}前后相机掉头约{round(turnaround['yaw_delta_deg'])}度，把掉头前后看到的东西放进同一个空间来比较。"})
            claims.append(self._claim("register", "camera_motion", f"相机在{self._viewref(turnaround['view_id'])}附近掉头约{turnaround['yaw_delta_deg']}度。", turnaround, [turnaround["view_id"]], role="transform"))
        sentences.append({"role": "conclusion", "text": f"这样接起来看，以出发朝向为参照，{ns}在{nr}的{RELATION_ZH[fact['relation']]}。"})
        claims.append(self._claim("relation", "relation", f"{ns}在{nr}的{RELATION_ZH[fact['relation']]}。", {k: fact[k] for k in ("relation", "dominant_delta_m", "margin_m")}, fact["subject_views"][:1] + fact["reference_views"][:1], role="conclusion"))
        return Round(
            round_index=4,
            turn_id="r04-crossview-relation",
            capability={"primary": "CR", "supporting": ["SR", "PT"], "sense_nova_subtask": "non-co-visible cross-view relation"},
            signature="G+G->F*->B_global->R->V",
            new_view_ids=new_views,
            evidence_view_ids=sorted(set(fact["subject_views"][:2] + fact["reference_views"][:2]), key=self.catalog.view_ids.index),
            nodes=[
                self._node("ground_s", "G", [s], "entity", s),
                self._node("ground_r", "G", [r], "entity", r),
                self._node("register", "F", ["camera_chain"], "transform_set", "pose_chain"),
                self._node("belief", "B", ["ground_s", "ground_r", "register"], "global_belief", [s, r]),
                self._node("relation", "R", ["belief"], "relation", fact["relation"]),
                self._node("verify", "V", ["relation"], "boolean", True),
            ],
            claims=claims,
            question_zh=f"到目前为止，{ns}和{nr}有没有在同一个视角里同时出现过？综合前后所有观察，{ns}在{nr}的什么方向？",
            answer_sentences=sentences,
            answer_key={"co_visible": False, "relation": fact["relation"]},
            diagnostics={"margin_m": fact["margin_m"]},
        )

    def _round_perspective(self, prefix: list[str], new_views: list[str]) -> Round | None:
        labels = list(self.catalog.entities)
        best = None
        for o in labels:
            for f in labels:
                if o == f:
                    continue
                shared = [
                    v
                    for v in prefix
                    if v in self.catalog.entities[o].views and v in self.catalog.entities[f].views
                    and self.catalog.visible_pixels(o, v) >= MIN_PIXELS and self.catalog.visible_pixels(f, v) >= MIN_PIXELS
                ]
                if not shared:
                    continue
                for t in labels:
                    if t in (o, f):
                        continue
                    tv = [v for v in prefix if v in self.catalog.entities[t].views]
                    if not tv or self.catalog.well_visible(t, tv) is None:
                        continue
                    fact = self.catalog.object_anchored(o, f, t)
                    if not fact["hard_quadrant_ok"]:
                        continue
                    cross_view = not (set(self.catalog.entities[t].views) & set(shared))
                    score = (1 if cross_view else 0, fact["angle_margin_deg"])
                    if best is None or score > best[0]:
                        fact["covis_view"] = shared[0]
                        fact["target_view"] = tv[0]
                        fact["cross_view"] = cross_view
                        best = (score, fact)
        if best is None:
            return None
        fact = best[1]
        no_, nf, nt = (self.catalog.entities[k].name for k in (fact["origin"], fact["facing"], fact["target"]))
        quad = QUADRANT_ZH[fact["quadrant"]]
        return Round(
            round_index=5,
            turn_id="r05-object-perspective",
            capability={"primary": "PT", "supporting": ["SR"], "sense_nova_subtask": "object-anchored perspective taking"},
            signature="G->B->F_query->P->R->V",
            new_view_ids=new_views,
            evidence_view_ids=sorted({fact["covis_view"], fact["target_view"]}, key=self.catalog.view_ids.index),
            nodes=[
                self._node("belief", "B", [fact["origin"], fact["facing"], fact["target"]], "global_belief", [fact["origin"], fact["facing"], fact["target"]]),
                self._node("query_frame", "F", [fact["origin"], fact["facing"]], "transform", {"origin": fact["origin"], "facing": fact["facing"]}),
                self._node("predict", "P", ["belief", "query_frame"], "query_coordinates", fact["query_xy_m"]),
                self._node("relation", "R", ["predict"], "relation", fact["quadrant"]),
                self._node("verify", "V", ["relation"], "boolean", fact["hard_quadrant_ok"]),
            ],
            claims=[
                self._claim("belief", "grounding", f"{no_}和{nf}在{self._viewref(fact['covis_view'])}同框；{nt}出现在{self._viewref(fact['target_view'])}。", [fact["origin"], fact["facing"], fact["target"]], [fact["covis_view"], fact["target_view"]], role="cue"),
                self._claim("query_frame", "frame", f"以{no_}指向{nf}的方向作为新的正前方。", {"origin": fact["origin"], "facing": fact["facing"]}, [fact["covis_view"]], role="transform"),
                self._claim("relation", "relation", f"在这个朝向下，{nt}位于{quad}。", fact, [fact["covis_view"], fact["target_view"]], role="conclusion"),
            ],
            question_zh=f"想象你走到{no_}旁边、面向{nf}站定；凭你目前对这个空间的印象，{nt}会在你的哪个方位？",
            answer_sentences=[
                {"role": "evidence", "text": f"{no_}和{nf}曾在{self._viewref(fact['covis_view'])}同框，{nt}则出现在{self._viewref(fact['target_view'])}。"},
                {"role": "transform", "text": f"把自己放到{no_}的位置、以朝向{nf}为正前方，再把{nt}的位置换算到这个朝向下。"},
                {"role": "conclusion", "text": f"{nt}会在你的{quad}。"},
            ],
            answer_key={"quadrant": fact["quadrant"]},
            diagnostics={"angle_margin_deg": fact["angle_margin_deg"], "cross_view": fact["cross_view"]},
        )

    def _round_epistemic_loop(self, prefix: list[str], new_views: list[str]) -> Round | None:
        if not self.catalog.loop_closed():
            return None
        unobserved = self.catalog.unobserved_named[0] if self.catalog.unobserved_named else None
        v0, v_last = self.catalog.view_ids[0], self.catalog.view_ids[-1]
        anchor = next(
            (label for label in sorted(self.catalog.entities) if v0 in self.catalog.entities[label].views and v_last in self.catalog.entities[label].views and self.catalog.visible_pixels(label, v_last) >= MIN_PIXELS),
            None,
        )
        if anchor is None or unobserved is None:
            return None
        nu = CATEGORY_NAMES_ZH[unobserved]
        na = self.catalog.entities[anchor].name
        return Round(
            round_index=6,
            turn_id="r06-epistemic-loop",
            capability={"primary": "CR", "supporting": ["PT"], "sense_nova_subtask": "epistemic boundary + loop persistence"},
            signature="G->B->V_unknown + F_loop->B->V",
            new_view_ids=new_views,
            evidence_view_ids=[v0, v_last],
            nodes=[
                self._node("search", "G", ["all_views", unobserved], "entity_or_none", None),
                self._node("verify_unknown", "V", ["search"], "unknown", "unknown"),
                self._node("loop", "F", [v0, v_last], "pose_equal", True),
                self._node("reobserve", "G", [v_last, anchor], "entity", anchor),
                self._node("verify_persist", "V", ["loop", "reobserve"], "boolean", True),
            ],
            claims=[
                self._claim("search", "never_observed", f"已给出的画面中从未出现{nu}。", None, list(self.catalog.view_ids), role="cue"),
                self._claim("verify_unknown", "epistemic", "画面里没出现不代表场景里没有，这一点无法确定。", "unknown", [], role="conclusion"),
                self._claim("loop", "loop_closure", "最后一个视角回到了与第1个视角相同的位置和朝向。", True, [v0, v_last], role="transform"),
                self._claim("reobserve", "persistence", f"{na}仍出现在原来的位置。", anchor, [v_last], role="conclusion"),
            ],
            question_zh=f"现在回到了起点。两个问题：到目前为止你见过{nu}吗？{na}还在最初看到的位置吗？",
            answer_sentences=[
                {"role": "evidence", "text": f"给出的画面里我从未见过{nu}。"},
                {"role": "calibration", "text": "不过这只能说明已给画面里没有，无法确定整个场景里有没有。"},
                {"role": "conclusion", "text": f"最后这个视角与第1个视角的位置和朝向一致，{na}仍在最初看到的位置。"},
            ],
            answer_key={"unobserved": "unknown", "anchor_persistent": True},
        )

    def _round_layout(self, prefix: list[str]) -> Round | None:
        facts = []
        used: set[str] = set()
        labels = list(self.catalog.entities)
        pairs = []
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                fact = self.catalog.relation(a, b)
                if fact["margin_m"] >= 0.5:
                    pairs.append(fact)
        pairs.sort(key=lambda f: -f["margin_m"])
        subject_uses: dict[str, int] = {}
        relation_uses: dict[str, int] = {}
        for fact in pairs:
            if fact["subject"] in used and fact["reference"] in used:
                continue
            if subject_uses.get(fact["subject"], 0) >= 1 or relation_uses.get(fact["relation"], 0) >= 2:
                continue
            facts.append(fact)
            used.update((fact["subject"], fact["reference"]))
            subject_uses[fact["subject"]] = subject_uses.get(fact["subject"], 0) + 1
            relation_uses[fact["relation"]] = relation_uses.get(fact["relation"], 0) + 1
            if len(facts) >= 3:
                break
        if len(facts) < 2:
            return None
        parts = [
            f"{self.catalog.entities[f['subject']].name}在{self.catalog.entities[f['reference']].name}的{RELATION_ZH[f['relation']]}"
            for f in facts
        ]
        text = "以出发朝向为参照：" + "；".join(parts) + "。"
        claims = [
            self._claim("layout", "relation", parts[i] + "。", facts[i], [], role="conclusion")
            for i in range(len(facts))
        ]
        return Round(
            round_index=7,
            turn_id="r07-layout-summary",
            capability={"primary": "CR", "supporting": ["SR"], "sense_nova_subtask": "layout synthesis from committed state"},
            signature="B_state->R^n->V",
            new_view_ids=[],
            evidence_view_ids=[],
            nodes=[self._node("state", "B", ["all_views"], "global_belief", sorted(used)), self._node("reads", "R", ["state"], "relation_set", [f["relation"] for f in facts]), self._node("verify", "V", ["reads"], "boolean", True)],
            claims=claims,
            question_zh="不再看新的画面，请根据你已经建立的空间印象，用一两句话概括这些主要物体之间的位置关系。",
            answer_sentences=[{"role": "conclusion", "text": text}],
            answer_key={"relations": [{k: f[k] for k in ("subject", "reference", "relation")} for f in facts]},
        )

    # -- compile ------------------------------------------------------------
    def compile(self) -> dict[str, Any]:
        views = self.catalog.view_ids
        builders = [
            (self._round_inventory, True),
            (self._round_covis_relation, False),
            (self._round_temporal, False),
            (self._round_crossview, False),
            (self._round_perspective, False),
            (self._round_epistemic_loop, False),
            (self._round_layout, True),
        ]
        rounds: list[Round] = []
        skipped: list[str] = []
        pending: list[str] = []
        released: list[str] = []
        for slot_index, (builder, prefix_only) in enumerate(builders):
            slot_views = [views[i] for i in RELEASE_SLOTS[slot_index]]
            new_views = pending + slot_views
            prefix = released + new_views
            round_ = builder(prefix) if prefix_only else builder(prefix, new_views)  # type: ignore[call-arg]
            if round_ is None:
                skipped.append(builder.__name__.replace("_round_", ""))
                pending = new_views
                continue
            round_.new_view_ids = new_views
            round_.round_index = len(rounds) + 1
            released = prefix
            pending = []
            rounds.append(round_)
        if pending:
            skipped.append(f"unreleased_views:{len(pending)}")
        has_integration = any(r.turn_id.startswith(("r04", "r05")) for r in rounds)
        if len(rounds) < 4 or not has_integration:
            raise ScenePlanError(f"insufficient certified rounds: kept={len(rounds)} skipped={skipped}")
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "scene_id": self.catalog.episode["scene_id"],
            "family_id": self.catalog.family_id,
            "split_lock": self.catalog.split_group,
            "frame_contract": {"frame_id": FRAME_ID, "surface_zh": FRAME_SURFACE_ZH},
            "view_ids": views,
            "unnamed_labels": self.catalog.unnamed_labels,
            "skipped_rounds": skipped,
            "rounds": [self._round_payload(r) for r in rounds],
        }
        self._validate(artifact)
        return artifact

    @staticmethod
    def _round_payload(round_: Round) -> dict[str, Any]:
        return {
            "turn_id": round_.turn_id,
            "round_index": round_.round_index,
            "new_view_ids": round_.new_view_ids,
            "evidence_view_ids": round_.evidence_view_ids,
            "capability": round_.capability,
            "program": {"semantic_signature": round_.signature, "nodes": round_.nodes},
            "claim_sheet": {"claims": round_.claims},
            "question_zh": round_.question_zh,
            "answer_sentences": round_.answer_sentences,
            "answer_zh": round_.answer_zh,
            "answer_key": round_.answer_key,
            "diagnostics": round_.diagnostics,
        }

    def _validate(self, artifact: dict[str, Any]) -> None:
        released: list[str] = []
        for round_ in artifact["rounds"]:
            released.extend(round_["new_view_ids"])
            for view_id in round_["evidence_view_ids"]:
                if view_id not in released:
                    raise ScenePlanError(f"{round_['turn_id']}: evidence {view_id} outside released prefix")
            surface = round_["question_zh"] + round_["answer_zh"]
            for token in FORBIDDEN_SURFACE_TOKENS:
                if token in surface:
                    raise ScenePlanError(f"{round_['turn_id']}: forbidden token {token!r} in surface")
            if UUID_RE.search(surface):
                raise ScenePlanError(f"{round_['turn_id']}: uuid leaked into surface")


def export_rgb(bundle_root: Path, out_dir: Path, view_ids: list[str]) -> dict[str, str]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for view_id in view_ids:
        target = out_dir / f"{view_id}.png"
        if not target.exists():
            with np.load(Path(bundle_root) / "views" / f"{view_id}.sensors.npz") as arrays:
                Image.fromarray(arrays["rgb"]).save(target)
        paths[view_id] = str(target)
    return paths


def to_sft_records(artifact: dict[str, Any], rgb_paths: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    system = (
        "你正在观察同一真实场景中按顺序释放的一组视角。请只依据已给出的画面维护空间信息并回答问题；"
        "证据不足时必须明确说无法确定。" + FRAME_SURFACE_ZH
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for round_ in artifact["rounds"]:
        content: list[dict[str, Any]] = []
        for view_id in round_["new_view_ids"]:
            content.append({"type": "image", "image": rgb_paths[view_id]})
        content.append({"type": "text", "text": round_["question_zh"]})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": round_["answer_zh"]})
    episode_record = {
        "record_id": f"scene-dialogue-{artifact['scene_id'][:12]}",
        "format": "qwen_multimodal_chat",
        "sample_type": "incremental_dialogue",
        "scene_id": artifact["scene_id"],
        "family_id": artifact["family_id"],
        "split_lock": artifact["split_lock"],
        "messages": messages,
        "loss_policy": {"train_on": "assistant_only"},
        "hidden_meta": {
            "semantic_signatures": [r["program"]["semantic_signature"] for r in artifact["rounds"]],
            "turn_ids": [r["turn_id"] for r in artifact["rounds"]],
        },
    }
    isolated = []
    for round_ in artifact["rounds"]:
        evidence = round_["evidence_view_ids"] or round_["new_view_ids"]
        if not evidence:
            continue
        content = [{"type": "image", "image": rgb_paths[v]} for v in evidence]
        content.append({"type": "text", "text": "以下画面来自同一场景。" + round_["question_zh"]})
        isolated.append(
            {
                "record_id": f"scene-isolated-{artifact['scene_id'][:12]}-{round_['turn_id']}",
                "format": "qwen_multimodal_chat",
                "sample_type": "isolated_qa",
                "scene_id": artifact["scene_id"],
                "family_id": artifact["family_id"],
                "split_lock": artifact["split_lock"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": round_["answer_zh"]},
                ],
                "loss_policy": {"train_on": "assistant_only"},
                "hidden_meta": {"turn_id": round_["turn_id"], "comparison_of": episode_record["record_id"]},
            }
        )
    return episode_record, isolated


def compile_sweep(
    release_index: Path,
    sweep_root: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    index = _read_json(release_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    balance: dict[str, int] = {}
    report: dict[str, Any] = {"compiled": [], "rejected": {}, "direction_balance": balance, "unnamed_labels": {}}
    episode_rows, isolated_rows = [], []
    count = 0
    for record in index["records"]:
        if record["automated_release_tier"] != "candidate":
            continue
        if limit is not None and count >= limit:
            break
        bundle = sweep_root / record["bundle_relpath"]
        name = record["acquisition_id"]
        try:
            compiler = SceneDialogueCompiler(bundle, direction_balance=balance)
            artifact = compiler.compile()
        except ScenePlanError as error:
            report["rejected"][name] = str(error)
            continue
        artifact["acquisition_id"] = name
        artifact["split"] = record["split"]
        rgb_paths = export_rgb(bundle, output_dir / "rgb" / name, artifact["view_ids"])
        scene_dir = output_dir / "scenes"
        scene_dir.mkdir(exist_ok=True)
        (scene_dir / f"{name}.dialogue.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        episode, isolated = to_sft_records(artifact, rgb_paths)
        episode["split"] = record["split"]
        for row in isolated:
            row["split"] = record["split"]
        episode_rows.append(episode)
        isolated_rows.extend(isolated)
        if artifact["unnamed_labels"]:
            report["unnamed_labels"][name] = artifact["unnamed_labels"]
        report["compiled"].append(
            {
                "acquisition_id": name,
                "split": record["split"],
                "rounds": [r["turn_id"] for r in artifact["rounds"]],
                "skipped": artifact["skipped_rounds"],
            }
        )
        count += 1
    for path, rows in (
        (output_dir / "train.dialogue_episode_sft.jsonl", episode_rows),
        (output_dir / "train.dialogue_isolated_sft.jsonl", isolated_rows),
    ):
        with path.open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "corpus_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report

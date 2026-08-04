"""Class-specific deterministic planners for T3/T4/T7/T8/T10.

These planners do not ask an LLM to invent facts or prose.  They execute a
small spatial program, attach a certificate to each claim, and then render a
human-readable cue -> transform -> conclusion answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from episode3d.scene_llm_pipeline.lexicon import QUADRANT_ZH, RELATION_ZH

from .catalog import (
    MIN_PIXELS,
    CatalogError,
    Entity,
    TrajectoryCatalog,
    circular_delta_deg,
)
from .contracts import SCHEMA_VERSION, ContractError, RoundBuilder, validate_artifact

FRAME_SURFACE_ZH = (
    "涉及房间平面方向时，以第1个视角的拍摄位置为原点、当时镜头朝向为前方，"
    "右手边为右；涉及假想站位时，以题目指定的面向对象作为新的前方。"
)
MIN_RELATION_MARGIN_M = 0.4


class PlanError(RuntimeError):
    """A passed acquisition cannot support a requested dialogue template."""


class BasePlanner:
    task_scope = "trajectory_reasoning"

    def __init__(
        self,
        bundle_root: Path,
        direction_balance: dict[str, int] | None = None,
    ) -> None:
        try:
            self.c = TrajectoryCatalog(bundle_root)
        except CatalogError as error:
            raise PlanError(str(error)) from error
        self.balance = direction_balance if direction_balance is not None else {}

    def ref(self, entity: Entity, view_id: str | None = None) -> str:
        try:
            return self.c.reference(entity, view_id)
        except CatalogError as error:
            raise PlanError(str(error)) from error

    def local_ref(self, entity: Entity, view_id: str) -> str:
        """Ground an ambiguous object without redundantly repeating current view."""
        return self.ref(entity, view_id).replace(self.view_ref(view_id), "")

    def view_num(self, view_id: str) -> int:
        return self.c.view_ids.index(view_id) + 1

    def view_ref(self, view_id: str) -> str:
        return f"第{self.view_num(view_id)}个视角"

    def inventory(self, view_id: str, new_views: list[str], turn_id: str) -> RoundBuilder:
        entities = self.c.named_visible(view_id)
        labels: list[str] = []
        source_ids: list[str] = []
        for entity in entities:
            if entity.name in labels:
                continue
            labels.append(entity.name)
            source_ids.append(entity.source_id)
            if len(labels) == 6:
                break
        if not labels:
            r = RoundBuilder(
                turn_id=turn_id,
                new_view_ids=new_views,
                evidence_view_ids=[view_id],
                capability={
                    "primary": "PT",
                    "supporting": ["B"],
                    "sense_nova_subtask": "initial observation-frame commit",
                },
                semantic_signature="G_scene->F_initial->B_partial->V",
                question_zh=f"先观察{self.view_ref(view_id)}。暂时不用勉强列出具体物体，请把这个站位和朝向记作后续比较的起始参照。",
                answer_key={"initial_frame_committed": True, "named_grounding_available": False},
            )
            r.node("g1", "G", ["view", "scene_envelope"], "grounding_scene", "observed_scene")
            r.node("f1", "F", ["camera_pose"], "frame_initial", view_id, ["g1"])
            r.node(
                "b1",
                "B",
                ["prior_belief", "scene_grounding", "frame"],
                "belief_partial",
                {"initial_view": view_id},
                ["g1", "f1"],
            )
            r.node("v1", "V", ["belief", "view_certificate"], "boolean", True, ["b1"])
            claim = r.claim(
                "c1",
                "f1",
                "frame_commit",
                "第1个视角被记作后续比较的起始站位与朝向。",
                view_id,
                [view_id],
                "conclusion",
            )
            r.sentence(
                "conclusion",
                "好的，我先把第1个视角作为起始参照；后面只根据新图更新，不会凭类别名称猜物体。",
                [claim],
            )
            return r
        listed = "、".join(labels)
        r = RoundBuilder(
            turn_id=turn_id,
            new_view_ids=new_views,
            evidence_view_ids=[view_id],
            capability={
                "primary": "SR",
                "supporting": ["G"],
                "sense_nova_subtask": "visual grounding",
            },
            semantic_signature="G(view,*)->B_local->V",
            question_zh=f"先观察{self.view_ref(view_id)}。你能辨认出哪些主要家具或设备？",
            answer_key={"visible_categories_zh": labels},
        )
        r.node("g1", "G", ["view", "referent_set"], "entity_set", source_ids)
        r.node("b1", "B", ["prior_belief", "entity_set"], "belief_local", source_ids, ["g1"])
        r.node("v1", "V", ["belief", "instance_masks"], "boolean", True, ["b1"])
        claim = r.claim(
            "c1", "g1", "grounding", f"图中可辨认出{listed}。", source_ids, [view_id], "conclusion"
        )
        r.sentence("conclusion", f"我能辨认出{listed}。", [claim])
        return r

    def _balanced_relation(
        self, candidates: list[tuple[Entity, Entity, dict[str, Any], str, str]]
    ) -> tuple[Entity, Entity, dict[str, Any], str, str] | None:
        if not candidates:
            return None
        candidates.sort(
            key=lambda row: (
                self.balance.get(row[2]["relation"], 0),
                -float(row[2]["margin_m"]),
            )
        )
        chosen = candidates[0]
        relation = chosen[2]["relation"]
        self.balance[relation] = self.balance.get(relation, 0) + 1
        return chosen

    def artifact(self, rounds: list[RoundBuilder], task_scope: str | None = None) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scene_id": self.c.scene_id,
            "family_id": self.c.family_id,
            "split_lock": self.c.split_group,
            "trajectory_class": self.c.trajectory_class,
            "task_scope": task_scope or self.task_scope,
            "frame_contract": {"surface_zh": FRAME_SURFACE_ZH},
            "source_bundle": str(self.c.root),
            "view_ids": self.c.view_ids,
            "rounds": [round_.payload(index + 1) for index, round_ in enumerate(rounds)],
        }
        try:
            validate_artifact(payload)
        except ContractError as error:
            raise PlanError(str(error)) from error
        return payload


class T3Planner(BasePlanner):
    """In-place observer rotation; no depth or parallax questions."""

    def compile(self) -> dict[str, Any]:
        v = self.c.view_ids
        if len(v) != 6 or not self.c.plan.get("depth_questions_forbidden"):
            raise PlanError("T3 requires six views and depth_questions_forbidden")
        rounds = [self.inventory(v[0], [v[0]], "t3-r01-ground-start")]

        common_candidates = []
        for first_view, later_views in ((v[0], [v[2], v[1]]), (v[1], [v[2]])):
            for entity in self.c.named_visible(first_view):
                later_view = self.c.well_visible(entity, later_views)
                if later_view is not None:
                    common_candidates.append((entity, first_view, later_view))
        anchor_info = common_candidates[0] if common_candidates else None
        yaws = [float(x) for x in self.c.plan["yaw_sequence_deg"]]
        turn = circular_delta_deg(yaws[0], yaws[2])
        r2 = RoundBuilder(
            turn_id="t3-r02-observer-rotation",
            new_view_ids=[v[1], v[2]],
            evidence_view_ids=[v[0], v[1], v[2]],
            capability={
                "primary": "PT",
                "supporting": ["SR"],
                "sense_nova_subtask": "observer rotation / situation transformation",
            },
            semantic_signature="G_track->F_rotation->B_identity->V_zero_translation",
            question_zh="连续看到第2、3个视角后，画面变化主要来自向前移动，还是来自站在原地转动观察方向？你依据什么判断？",
            answer_key={"motion_type": "in_place_rotation", "yaw_change_deg": round(turn, 3)},
            diagnostics={
                "maximum_translational_baseline_m": self.c.quality["gates"]["T3"][
                    "maximum_translational_baseline_m"
                ]
            },
        )
        if anchor_info is None:
            r2.node(
                "g1",
                "G",
                ["view_sequence", "scene_envelope"],
                "grounding_scene",
                "rotating_scene_views",
            )
        else:
            anchor, anchor_first_view, anchor_later_view = anchor_info
            r2.node(
                "g1",
                "G",
                ["view_sequence", "entity"],
                "track",
                {"entity": anchor.source_id, "views": [anchor_first_view, anchor_later_view]},
            )
        r2.node(
            "f1",
            "F",
            ["camera_pose_sequence"],
            "motion_rotation",
            {"translation_m": 0.0, "yaw_change_deg": round(turn, 3)},
            ["g1"],
        )
        persistence_value = (
            {"persistent_scene": True}
            if anchor_info is None
            else {"persistent": anchor_info[0].source_id}
        )
        r2.node(
            "b1",
            "B",
            ["prior_belief", "grounding", "transform"],
            "identity_belief",
            persistence_value,
            ["g1", "f1"],
        )
        r2.node("v1", "V", ["motion", "pose_certificate"], "boolean", True, ["b1"])
        if anchor_info is None:
            c21 = r2.claim(
                "c1",
                "g1",
                "scene_cue",
                "连续画面展示同一空间的不同朝向。",
                "same_scene",
                v[:3],
                "cue",
            )
            cue_text = "连续画面中的墙面与空间边界随视线方向成组换位，而不是产生向前接近的视差。"
        else:
            anchor, anchor_first_view, anchor_later_view = anchor_info
            anchor_ref = self.ref(anchor, anchor_first_view)
            c21 = r2.claim(
                "c1",
                "g1",
                "tracking",
                f"{anchor_ref}在相邻转向视角中仍可对应。",
                anchor.source_id,
                [anchor_first_view, anchor_later_view],
                "cue",
            )
            cue_text = f"例如{anchor_ref}能在相邻转向视角中继续对应上。"
        c22 = r2.claim(
            "c2",
            "f1",
            "camera_motion",
            f"三个视角拍摄位置相同，朝向累计改变约{round(turn):.0f}度。",
            {"translation_m": 0.0, "yaw_change_deg": round(turn, 3)},
            v[:3],
            "transform",
        )
        c23 = r2.claim(
            "c3",
            "v1",
            "causal_attribution",
            "画面变化来自观察者原地转向，而不是场景物体移动。",
            "in_place_rotation",
            v[:3],
            "conclusion",
        )
        r2.sentence("cue", cue_text, [c21])
        r2.sentence(
            "transform",
            f"这三张图的拍摄点没有平移，只是观察方向累计改变了约{round(turn):.0f}度。",
            [c22],
        )
        r2.sentence(
            "conclusion",
            "所以主要是观察者在原地转身，不能把画面位置变化理解成场景物体移动。",
            [c23],
        )
        rounds.append(r2)

        early = v[:3]
        later = v[3:5]
        rear = [
            e
            for e in self.c.entities_by_source.values()
            if self.c.well_visible(e, early) is None and self.c.well_visible(e, later) is not None
        ]
        if rear:
            rear.sort(key=lambda e: -max(self.c.pixels(e, x) for x in later))
            target = rear[0]
            target_view = self.c.well_visible(target, later)
            assert target_view is not None
            target_ref = self.ref(target, target_view)
            r3 = RoundBuilder(
                turn_id="t3-r03-rear-reveal",
                new_view_ids=[v[3], v[4]],
                evidence_view_ids=[*early, *later],
                capability={
                    "primary": "CR",
                    "supporting": ["PT", "SR"],
                    "sense_nova_subtask": "panoramic scene reconstruction",
                },
                semantic_signature="G_new->F_rotation->B_update->V_persistence",
                question_zh=f"转到第4、5个视角后才看到{target_ref}。这更像是它刚被放进房间，还是原先位于视野外？",
                answer_key={"belief_update": "previously_out_of_view", "entity": target.source_id},
            )
            r3.node(
                "g1",
                "G",
                ["view_sequence", "entity"],
                "visibility_track",
                {"early": [], "later": [target_view]},
            )
            r3.node(
                "f1",
                "F",
                ["camera_pose_sequence"],
                "transform_rotation",
                {"stationary": True},
                ["g1"],
            )
            r3.node(
                "b1",
                "B",
                ["prior_belief", "visibility", "transform"],
                "belief_scene_state",
                {"entity": target.source_id, "status": "observed_rear_sector"},
                ["g1", "f1"],
            )
            r3.node("v1", "V", ["belief", "visibility_certificate"], "boolean", True, ["b1"])
            c31 = r3.claim(
                "c1",
                "g1",
                "visibility",
                f"前三个视角没有可靠看到{target.name}，{self.view_ref(target_view)}开始能看清它。",
                {"early": [], "first_seen": target_view},
                [*early, *later],
                "cue",
            )
            c32 = r3.claim(
                "c2",
                "f1",
                "frame_change",
                "相机一直在同一位置，只是转向了房间另一侧。",
                "rotation_only",
                [*early, *later],
                "transform",
            )
            c33 = r3.claim(
                "c3",
                "b1",
                "persistence",
                f"{target.name}原先在视野外，不是中途出现的新物体。",
                "previously_out_of_view",
                [target_view],
                "conclusion",
            )
            r3.sentence(
                "cue",
                f"前三个视角里没有可靠看到它，到了{self.view_ref(target_view)}才看清{target_ref}。",
                [c31],
            )
            r3.sentence("transform", "这一段相机仍在原地，只是继续转向房间另一侧。", [c32])
            r3.sentence(
                "conclusion", "因此更合理的判断是它原先位于视野外，而不是刚被放进房间。", [c33]
            )
        else:
            half_turn = circular_delta_deg(yaws[0], yaws[3])
            r3 = RoundBuilder(
                turn_id="t3-r03-opposite-sector",
                new_view_ids=[v[3], v[4]],
                evidence_view_ids=[*early, *later],
                capability={
                    "primary": "PT",
                    "supporting": ["CR", "MM"],
                    "sense_nova_subtask": "opposite-sector scene reconstruction",
                },
                semantic_signature="G_scene->F_half_turn->P_opposite_sector->B_update->V",
                question_zh="转到第4、5个视角后，当前观察方向与起始方向是什么关系？画面内容明显不同是否足以说明场景物体移动了？",
                answer_key={"view_004_relative_to_start": "opposite", "objects_moved": False},
            )
            r3.node(
                "g1",
                "G",
                ["view_sequence", "scene_envelope"],
                "grounding_scene",
                "early_and_opposite_sectors",
            )
            r3.node(
                "f1",
                "F",
                ["camera_pose_sequence"],
                "transform_rotation",
                {"half_turn_deg": round(half_turn, 3)},
                ["g1"],
            )
            r3.node(
                "p1",
                "P",
                ["belief_partial", "target_view_frame"],
                "prediction_opposite_sector",
                True,
                ["f1"],
            )
            r3.node(
                "b1",
                "B",
                ["prior_belief", "prediction", "new_observation"],
                "belief_scene_state",
                {"opposite_sector_observed": True},
                ["p1"],
            )
            r3.node("v1", "V", ["belief", "pose_certificate"], "boolean", True, ["b1"])
            c31 = r3.claim(
                "c1",
                "g1",
                "scene_change",
                "第4、5个视角展示了起始画面之外的另一片空间。",
                "new_sector",
                [*early, *later],
                "cue",
            )
            c32 = r3.claim(
                "c2",
                "f1",
                "half_turn",
                f"第4个视角相对起始方向转过约{round(half_turn):.0f}度。",
                round(half_turn, 3),
                [v[0], v[3]],
                "transform",
            )
            c33 = r3.claim(
                "c3",
                "b1",
                "persistence",
                "内容变化由观察方向改变解释，不支持物体移动。",
                False,
                [*early, *later],
                "conclusion",
            )
            r3.sentence("cue", "第4、5个视角展示了起始画面之外的另一片空间。", [c31])
            r3.sentence(
                "transform",
                f"其中第4个视角相对起始方向已转过约{round(half_turn):.0f}度，接近正对起始方向的反向。",
                [c32],
            )
            r3.sentence(
                "conclusion",
                "因此画面内容不同是换了观察方向的预期结果，不能据此说场景物体移动了。",
                [c33],
            )
        rounds.append(r3)

        relation = self._t3_cross_relation(v)
        if relation is None:
            rounds.append(self._t3_rotation_closure(v, yaws))
            return self.artifact(rounds)
        front, back, fact, front_view, back_view = relation
        nf, nb = self.ref(front, front_view), self.ref(back, back_view)
        r4 = RoundBuilder(
            turn_id="t3-r04-panorama-relation",
            new_view_ids=[v[5]],
            evidence_view_ids=sorted({front_view, back_view, v[5]}, key=v.index),
            capability={
                "primary": "CR",
                "supporting": ["PT", "SR"],
                "sense_nova_subtask": "non-co-visible panoramic relation",
            },
            semantic_signature="G_front+G_rear->F_pose_chain->B_global->R->V",
            question_zh=f"现在六个方向都观察过了。{nf}和{nb}是否需要跨视角才能联系起来？以第1个视角的朝向为前方，{nf}在{nb}的什么方向？",
            answer_key={"cross_view": True, "relation": fact["relation"]},
            diagnostics={"relation_margin_m": fact["margin_m"]},
        )
        r4.node(
            "g1",
            "G",
            ["view", "entity"],
            "entity",
            {"source_id": front.source_id, "view": front_view},
        )
        r4.node(
            "g2",
            "G",
            ["view", "entity"],
            "entity",
            {"source_id": back.source_id, "view": back_view},
        )
        r4.node(
            "f1",
            "F",
            ["camera_pose_sequence", "first_view_frame"],
            "transform_set",
            "pose_chain",
            ["g1", "g2"],
        )
        r4.node(
            "b1",
            "B",
            ["entity", "entity", "transform"],
            "belief_global",
            [front.source_id, back.source_id],
            ["g1", "g2", "f1"],
        )
        r4.node("r1", "R", ["entity_pair", "first_view_frame"], "relation_direction", fact, ["b1"])
        r4.node(
            "v1",
            "V",
            ["relation", "geometry_margin"],
            "boolean",
            fact["margin_m"] >= MIN_RELATION_MARGIN_M,
            ["r1"],
        )
        c41 = r4.claim(
            "c1",
            "g1",
            "cross_view_cue",
            f"{nf}来自{self.view_ref(front_view)}，{nb}来自{self.view_ref(back_view)}。",
            [front.source_id, back.source_id],
            [front_view, back_view],
            "cue",
        )
        c42 = r4.claim(
            "c2",
            "f1",
            "registration",
            "把原地旋转的六个朝向注册到第1个视角的平面方向。",
            "first_view_frame",
            [front_view, back_view, v[5]],
            "transform",
        )
        c43 = r4.claim(
            "c3",
            "r1",
            "relation",
            f"{front.name}在{back.name}的{RELATION_ZH[fact['relation']]}。",
            fact,
            [front_view, back_view],
            "conclusion",
        )
        r4.sentence(
            "cue",
            f"需要跨视角：{nf}主要由{self.view_ref(front_view)}提供线索，{nb}主要由{self.view_ref(back_view)}提供线索。",
            [c41],
        )
        r4.sentence(
            "transform", "把这些原地转向的观察放回第1个视角所定义的同一张平面图后再比较。", [c42]
        )
        r4.sentence("conclusion", f"{nf}在{nb}的{RELATION_ZH[fact['relation']]}。", [c43])
        rounds.append(r4)
        return self.artifact(rounds)

    def _t3_rotation_closure(self, views: list[str], yaws: list[float]) -> RoundBuilder:
        gap = circular_delta_deg(yaws[-1], yaws[0])
        r = RoundBuilder(
            turn_id="t3-r04-rotation-closure",
            new_view_ids=[views[-1]],
            evidence_view_ids=views,
            capability={
                "primary": "CR",
                "supporting": ["PT", "MM"],
                "sense_nova_subtask": "rotation sequence composition and closure",
            },
            semantic_signature="F_rotation_chain->M_net_heading->R_closure->V",
            question_zh="六个朝向都看完后，最后一张图是否已经与第1张回到完全相同的朝向？请区分“采样覆盖了一圈”和“最后姿态真正闭合”。",
            answer_key={
                "sampled_full_circle": True,
                "final_pose_closed": False,
                "remaining_heading_gap_deg": round(gap, 3),
            },
        )
        r.node("f1", "F", ["camera_pose_sequence"], "transform_set", yaws)
        r.node("m1", "M", ["transform_set"], "angle_gap_deg", round(gap, 3), ["f1"])
        r.node("r1", "R", ["pose_sequence", "closure_rule"], "closure_relation", False, ["m1"])
        r.node("v1", "V", ["closure", "pose_certificate"], "boolean", True, ["r1"])
        c1 = r.claim(
            "c1", "f1", "sequence", "六个视角按约60度间隔采样了不同朝向。", yaws, views, "cue"
        )
        c2 = r.claim(
            "c2",
            "m1",
            "heading_gap",
            f"最后朝向与起始朝向仍相差约{round(gap):.0f}度。",
            round(gap, 3),
            [views[0], views[-1]],
            "transform",
        )
        c3 = r.claim(
            "c3",
            "r1",
            "closure",
            "方向采样覆盖一圈，但最后姿态没有回到起始朝向。",
            False,
            views,
            "conclusion",
        )
        r.sentence("cue", "六张图按固定角度间隔依次观察，方向采样已经覆盖了房间四周。", [c1])
        r.sentence("transform", f"但最后一个朝向与起始朝向仍相差约{round(gap):.0f}度。", [c2])
        r.sentence("conclusion", "所以“覆盖了一圈”成立，“最后姿态完全闭合”不成立。", [c3])
        return r

    def _t3_cross_relation(
        self, views: list[str]
    ) -> tuple[Entity, Entity, dict[str, Any], str, str] | None:
        early = views[:3]
        late = views[3:]
        front_rows = [
            (e, self.c.well_visible(e, early)) for e in self.c.entities_by_source.values()
        ]
        back_rows = [
            (e, self.c.well_visible(e, [views[5], *late[:2]]))
            for e in self.c.entities_by_source.values()
        ]
        candidates = []
        for a, av in front_rows:
            if av is None:
                continue
            for b, bv in back_rows:
                if bv is None or a.source_id == b.source_id or a.label == b.label:
                    continue
                if set(self.c.visibility_timeline(a)) & set(self.c.visibility_timeline(b)):
                    continue
                fact = self.c.relation(a, b, views[0])
                if fact["margin_m"] >= MIN_RELATION_MARGIN_M:
                    candidates.append((a, b, fact, av, bv))
        return self._balanced_relation(candidates)


class T4Planner(BasePlanner):
    """Camera orbit around a persistent focus; never intrinsic orientation."""

    def compile(self) -> dict[str, Any]:
        v = self.c.view_ids
        if len(v) != 8 or self.c.plan.get("orientation_questions_allowed", True):
            raise PlanError("T4 requires eight views and disabled intrinsic orientation questions")
        focus = self.c.entity(self.c.selection["focus_entity_id"])
        if any(self.c.pixels(focus, view) < MIN_PIXELS for view in v):
            raise PlanError("T4 focus does not pass compiler pixel gate in every view")
        focus_ref = self.ref(focus, v[0])
        r1 = RoundBuilder(
            turn_id="t4-r01-bind-focus",
            new_view_ids=[v[0]],
            evidence_view_ids=[v[0]],
            capability={
                "primary": "SR",
                "supporting": ["G"],
                "sense_nova_subtask": "focus grounding",
            },
            semantic_signature="G_focus->B_identity->V",
            question_zh="接下来我会围绕同一件物体走动。先看第1个视角，画面中心附近的主要焦点是什么？",
            answer_key={"focus": focus.source_id, "category_zh": focus.name},
        )
        r1.node("g1", "G", ["view", "focus_hint"], "entity", focus.source_id)
        r1.node("b1", "B", ["prior_belief", "entity"], "identity_belief", focus.source_id, ["g1"])
        r1.node("v1", "V", ["identity", "instance_mask"], "boolean", True, ["b1"])
        c11 = r1.claim(
            "c1",
            "g1",
            "grounding",
            f"轨迹焦点是{focus_ref}。",
            focus.source_id,
            [v[0]],
            "conclusion",
        )
        r1.sentence(
            "conclusion", f"焦点是{focus_ref}。后续视角要继续把它当作同一件物体来跟踪。", [c11]
        )

        azimuths = [float(x) for x in self.c.plan["azimuth_deg_per_view"]]
        local_arc = circular_delta_deg(azimuths[0], azimuths[2])
        r2 = RoundBuilder(
            turn_id="t4-r02-orbit-identity",
            new_view_ids=[v[1], v[2]],
            evidence_view_ids=v[:3],
            capability={
                "primary": "PT",
                "supporting": ["SR", "CR"],
                "sense_nova_subtask": "viewpoint change with object persistence",
            },
            semantic_signature="G_track->F_orbit->B_identity->V",
            question_zh=f"看完第2、3个视角，{focus.name}是换成了另一件同类物体，还是观察者绕着同一件物体移动了？",
            answer_key={"same_instance": True, "observer_arc_deg": round(local_arc, 3)},
        )
        r2.node(
            "g1",
            "G",
            ["view_sequence", "entity"],
            "track",
            {"entity": focus.source_id, "views": v[:3]},
        )
        r2.node(
            "f1",
            "F",
            ["camera_pose_sequence", "focus_position"],
            "motion_orbit",
            {"arc_deg": round(local_arc, 3)},
            ["g1"],
        )
        r2.node(
            "b1",
            "B",
            ["track", "transform"],
            "identity_belief",
            {"same_instance": True},
            ["g1", "f1"],
        )
        r2.node("v1", "V", ["identity", "instance_masks"], "boolean", True, ["b1"])
        c21 = r2.claim(
            "c1",
            "g1",
            "tracking",
            f"同一{focus.name}在前三个视角持续可见。",
            focus.source_id,
            v[:3],
            "cue",
        )
        c22 = r2.claim(
            "c2",
            "f1",
            "camera_motion",
            f"观察位置沿目标周围移动约{round(local_arc):.0f}度。",
            round(local_arc, 3),
            v[:3],
            "transform",
        )
        c23 = r2.claim(
            "c3",
            "b1",
            "identity",
            "变化来自观察位置，不是替换或旋转物体。",
            True,
            v[:3],
            "conclusion",
        )
        r2.sentence("cue", f"{focus_ref}在三张图里都能连续对应，周围背景则随视点变化。", [c21])
        r2.sentence("transform", f"相机沿它周围移动了约{round(local_arc):.0f}度。", [c22])
        r2.sentence(
            "conclusion",
            "因此这是观察者绕同一物体移动，不是换了物体，也不能据此声称物体自身发生了旋转。",
            [c23],
        )

        pairs = self.c.quality["gates"]["T4"]["opposite_view_index_pairs"]
        usable = [(int(a), int(b)) for a, b in pairs if int(b) <= 4]
        if not usable:
            raise PlanError("T4: no certified opposite pair within third release")
        a, b = max(
            usable, key=lambda pair: circular_delta_deg(azimuths[pair[0]], azimuths[pair[1]])
        )
        opposite_arc = circular_delta_deg(azimuths[a], azimuths[b])
        r3 = RoundBuilder(
            turn_id="t4-r03-opposite-view",
            new_view_ids=[v[3], v[4]],
            evidence_view_ids=[v[a], v[b]],
            capability={
                "primary": "PT",
                "supporting": ["MM"],
                "sense_nova_subtask": "opposite viewpoint transformation",
            },
            semantic_signature="G_focus_pair->F_orbit->M_arc->P_opposite->V",
            question_zh=f"比较{self.view_ref(v[a])}和{self.view_ref(v[b])}：后一个视角是否已经接近从焦点物体的相对侧观察？注意回答的是相机站位，不是物体正反面。",
            answer_key={"near_opposite_side": True, "separation_deg": round(opposite_arc, 3)},
        )
        r3.node(
            "g1",
            "G",
            ["view_pair", "entity"],
            "track",
            {"entity": focus.source_id, "views": [v[a], v[b]]},
        )
        r3.node(
            "f1",
            "F",
            ["camera_pose_pair", "focus_position"],
            "transform_orbit",
            [azimuths[a], azimuths[b]],
            ["g1"],
        )
        r3.node("m1", "M", ["transform"], "angle_deg", round(opposite_arc, 3), ["f1"])
        r3.node(
            "p1",
            "P",
            ["belief_identity", "target_view_frame"],
            "prediction_opposite_side",
            True,
            ["m1"],
        )
        r3.node("v1", "V", ["prediction", "certified_opposite_pair"], "boolean", True, ["p1"])
        c31 = r3.claim(
            "c1",
            "g1",
            "identity",
            f"两个视角中的焦点都是同一{focus.name}。",
            focus.source_id,
            [v[a], v[b]],
            "cue",
        )
        c32 = r3.claim(
            "c2",
            "m1",
            "orbit_angle",
            f"两次相机站位绕目标相隔约{round(opposite_arc):.0f}度。",
            round(opposite_arc, 3),
            [v[a], v[b]],
            "transform",
        )
        c33 = r3.claim(
            "c3",
            "p1",
            "viewpoint",
            "后一个视角接近从相对侧观察。",
            True,
            [v[a], v[b]],
            "conclusion",
        )
        r3.sentence(
            "cue",
            f"两幅图都在跟踪同一个{focus.name}，但周围背景的相对位置明显换到了另一边。",
            [c31],
        )
        r3.sentence("transform", f"两次相机站位绕目标相隔约{round(opposite_arc):.0f}度。", [c32])
        r3.sentence(
            "conclusion",
            "所以后一个视角已经接近相对侧；这只描述相机绕行，不定义物体的正面或背面。",
            [c33],
        )

        coverage = float(self.c.quality["gates"]["T4"]["arc_coverage_deg"])
        complete = bool(self.c.quality["gates"]["T4"]["complete_orbit"])
        conclusion = "构成了完整环绕" if complete else "没有闭合成完整一圈"
        r4 = RoundBuilder(
            turn_id="t4-r04-arc-closure",
            new_view_ids=[v[5], v[6], v[7]],
            evidence_view_ids=v,
            capability={
                "primary": "CR",
                "supporting": ["PT", "MM"],
                "sense_nova_subtask": "path coverage and closure",
            },
            semantic_signature="B_orbit_path->M_coverage->R_closure->V",
            question_zh="全部八个视角看完后，这段轨迹完整绕了一圈吗？请区分“看到了很多侧面”和“轨迹真正闭合”。",
            answer_key={"complete_orbit": complete, "arc_coverage_deg": round(coverage, 3)},
        )
        r4.node("b1", "B", ["prior_belief", "camera_pose_sequence"], "belief_orbit_path", azimuths)
        r4.node("m1", "M", ["belief_path"], "angle_coverage_deg", round(coverage, 3), ["b1"])
        r4.node("r1", "R", ["belief_path", "closure_rule"], "closure_relation", complete, ["m1"])
        r4.node("v1", "V", ["closure", "trajectory_certificate"], "boolean", True, ["r1"])
        c41 = r4.claim(
            "c1",
            "b1",
            "path",
            "八个视角保持同一焦点并沿其周围连续移动。",
            focus.source_id,
            v,
            "cue",
        )
        c42 = r4.claim(
            "c2",
            "m1",
            "coverage",
            f"轨迹覆盖的方位弧约为{round(coverage):.0f}度。",
            round(coverage, 3),
            v,
            "transform",
        )
        c43 = r4.claim("c3", "r1", "closure", f"轨迹{conclusion}。", complete, v, "conclusion")
        r4.sentence(
            "cue",
            f"八张图始终围绕同一{focus.name}，但连续看见多个侧面并不自动等于回到起点。",
            [c41],
        )
        r4.sentence("transform", f"这段路径实际覆盖约{round(coverage):.0f}度。", [c42])
        r4.sentence("conclusion", f"因此它{conclusion}。", [c43])
        return self.artifact([r1, r2, r3, r4])


class T7Planner(BasePlanner):
    """Same planar station under a controlled camera-height intervention."""

    def compile(self) -> dict[str, Any]:
        v = self.c.view_ids
        heights = [float(x) for x in self.c.plan["height_sequence_m"]]
        pitches = [float(x) for x in self.c.plan["pitch_sequence_deg"]]
        if len(v) != 3 or len(heights) != 3 or len(pitches) != 3:
            raise PlanError("T7 requires three aligned height/pitch views")
        r1 = self.inventory(v[0], [v[0]], "t7-r01-low-view")
        common = [e for e in self.c.named_visible(v[0]) if self.c.pixels(e, v[1]) >= MIN_PIXELS]
        anchor = common[0] if common else None
        rise = heights[1] - heights[0]
        r2 = RoundBuilder(
            turn_id="t7-r02-height-intervention",
            new_view_ids=[v[1]],
            evidence_view_ids=[v[0], v[1]],
            capability={
                "primary": "PT",
                "supporting": ["SR", "MM"],
                "sense_nova_subtask": "viewpoint elevation transformation",
            },
            semantic_signature="G_track->F_height->M_delta->B_identity->V",
            question_zh="第2个视角与第1个相比，主要变化是观察者在地面上换了位置，还是视点在原处升高了？大约升高多少，场景里的物体是否因此移动？",
            answer_key={
                "same_planar_station": True,
                "height_rise_m": round(rise, 3),
                "objects_moved": False,
            },
        )
        if anchor is None:
            r2.node(
                "g1", "G", ["view_pair", "scene_envelope"], "grounding_scene", "same_scene_envelope"
            )
        else:
            r2.node(
                "g1",
                "G",
                ["view_pair", "entity"],
                "track",
                {"entity": anchor.source_id, "views": v[:2]},
            )
        r2.node(
            "f1",
            "F",
            ["camera_pose_pair"],
            "transform_height",
            {"same_xy": True, "height_delta_m": round(rise, 3)},
            ["g1"],
        )
        r2.node("m1", "M", ["transform"], "height_delta_m", round(rise, 3), ["f1"])
        r2.node(
            "b1",
            "B",
            ["prior_belief", "track", "transform"],
            "identity_belief",
            {"objects_static": True},
            ["g1", "f1"],
        )
        r2.node("v1", "V", ["belief", "pose_certificate"], "boolean", True, ["m1", "b1"])
        if anchor is None:
            c21 = r2.claim(
                "c1",
                "g1",
                "scene_tracking",
                "两个高度的图保持同一场景边界与平面站位。",
                "same_scene_envelope",
                v[:2],
                "cue",
            )
            cue_text = "两张图中的墙面和空间边界保持同一场景结构，只是观察高度改变了。"
        else:
            anchor_ref = self.ref(anchor, v[0])
            c21 = r2.claim(
                "c1",
                "g1",
                "tracking",
                f"{anchor_ref}在两个高度的图中仍是同一地标。",
                anchor.source_id,
                v[:2],
                "cue",
            )
            cue_text = f"例如{anchor_ref}仍能对应为同一地标，只是画面中的观察角度变了。"
        c22 = r2.claim(
            "c2",
            "f1",
            "height_change",
            f"相机平面站位不变，视点升高约{rise:.1f}米。",
            round(rise, 3),
            v[:2],
            "transform",
        )
        c23 = r2.claim(
            "c3",
            "b1",
            "persistence",
            "变化来自视点高度，场景物体没有移动。",
            False,
            v[:2],
            "conclusion",
        )
        r2.sentence("cue", cue_text, [c21])
        r2.sentence("transform", f"相机没有在地面上换站位，而是在原处升高了约{rise:.1f}米。", [c22])
        r2.sentence("conclusion", "所以变化来自视点升高，不能解释成场景里的物体移动。", [c23])

        r3 = self._high_view_round(v, heights, pitches)
        return self.artifact([r1, r2, r3])

    def _high_view_round(
        self, v: list[str], heights: list[float], pitches: list[float]
    ) -> RoundBuilder:
        visible = self.c.named_visible(v[2])
        candidates: list[tuple[Entity, Entity, dict[str, Any]]] = []
        for a in visible:
            for b in visible:
                if a.source_id == b.source_id:
                    continue
                fact = self.c.vertical_relation(a, b)
                if fact is not None:
                    candidates.append((a, b, fact))
        candidates.sort(key=lambda row: -float(row[2]["clearance_m"]))
        rise = heights[2] - heights[0]
        if candidates:
            a, b, fact = candidates[0]
            na, nb = self.ref(a, v[2]), self.ref(b, v[2])
            rel_zh = "上方" if fact["relation"] == "above" else "下方"
            r = RoundBuilder(
                turn_id="t7-r03-world-vertical-relation",
                new_view_ids=[v[2]],
                evidence_view_ids=v,
                capability={
                    "primary": "SR",
                    "supporting": ["PT", "MM"],
                    "sense_nova_subtask": "world vertical relation under high downward view",
                },
                semantic_signature="G_pair->F_height_pitch->R_vertical->M_margin->V",
                question_zh=f"第3个视角升到高处并向下看。请判断{na}相对{nb}在真实空间中更靠上还是更靠下；不要只按它们落在画面的上半或下半来猜。",
                answer_key={"vertical_relation": fact["relation"]},
                diagnostics={
                    "vertical_clearance_m": fact["clearance_m"],
                    "camera_rise_from_low_m": round(rise, 3),
                },
            )
            r.node("g1", "G", ["view", "entity_pair"], "entity_pair", [a.source_id, b.source_id])
            r.node(
                "f1",
                "F",
                ["camera_height", "camera_pitch", "world_vertical_frame"],
                "transform_height_pitch",
                {"height_m": heights[2], "pitch_deg": pitches[2]},
                ["g1"],
            )
            r.node(
                "r1",
                "R",
                ["entity_pair", "world_vertical_frame"],
                "relation_vertical",
                fact,
                ["f1"],
            )
            r.node(
                "m1",
                "M",
                ["relation", "object_extents"],
                "distance_clearance_m",
                fact["clearance_m"],
                ["r1"],
            )
            r.node("v1", "V", ["relation", "extent_margin"], "boolean", True, ["m1"])
            c1 = r.claim(
                "c1",
                "g1",
                "grounding",
                f"{na}与{nb}在高位视角可辨认。",
                [a.source_id, b.source_id],
                [v[2]],
                "cue",
            )
            c2 = r.claim(
                "c2",
                "f1",
                "viewpoint",
                f"第3个视角比低位升高约{rise:.1f}米，并向下俯视。",
                {"rise_m": round(rise, 3), "pitch_deg": pitches[2]},
                v,
                "transform",
            )
            c3 = r.claim(
                "c3",
                "r1",
                "vertical_relation",
                f"按世界竖直方向，{a.name}在{b.name}的{rel_zh}。",
                fact,
                [v[2]],
                "conclusion",
            )
            r.sentence("cue", f"在高位图中可以把{na}和{nb}分别绑定到具体物体。", [c1])
            r.sentence(
                "transform",
                f"但相机比低位升高了约{rise:.1f}米并向下俯视，所以不能把画面上下直接当成真实高低。",
                [c2],
            )
            r.sentence(
                "conclusion", f"按物体在真实空间中的竖直范围比较，{na}位于{nb}的{rel_zh}。", [c3]
            )
            return r

        anchor = next(
            (
                entity
                for entity in visible
                if all(self.c.pixels(entity, view) >= MIN_PIXELS for view in v)
            ),
            None,
        )
        name = self.ref(anchor, v[0]) if anchor is not None else "墙面与场景边界"
        r = RoundBuilder(
            turn_id="t7-r03-high-downward-view",
            new_view_ids=[v[2]],
            evidence_view_ids=v,
            capability={
                "primary": "PT",
                "supporting": ["CR", "MM"],
                "sense_nova_subtask": "height and pitch composition",
            },
            semantic_signature="G_track->F_height+pitch->P_appearance->B_identity->V",
            question_zh="第3个视角为什么更像高处俯视，而不是房间和家具整体向下移动了？",
            answer_key={"camera_intervention": "higher_and_downward", "objects_moved": False},
            diagnostics={"camera_rise_from_low_m": round(rise, 3), "pitch_deg": pitches[2]},
        )
        if anchor is None:
            r.node(
                "g1",
                "G",
                ["view_sequence", "scene_envelope"],
                "grounding_scene",
                "stable_scene_envelope",
            )
        else:
            r.node("g1", "G", ["view_sequence", "entity"], "track", anchor.source_id)
        r.node(
            "f1",
            "F",
            ["camera_height", "camera_pitch"],
            "transform_height_pitch",
            {"rise_m": round(rise, 3), "pitch_deg": pitches[2]},
            ["g1"],
        )
        r.node(
            "p1",
            "P",
            ["belief_identity", "target_view_frame"],
            "prediction_appearance_change",
            "high_downward",
            ["f1"],
        )
        r.node(
            "b1",
            "B",
            ["track", "prediction"],
            "identity_belief",
            {"objects_static": True},
            ["g1", "p1"],
        )
        r.node("v1", "V", ["belief", "pose_certificate"], "boolean", True, ["b1"])
        grounding_value: Any = "stable_scene_envelope" if anchor is None else anchor.source_id
        c1 = r.claim(
            "c1", "g1", "tracking", f"{name}在三幅图中保持同一场景结构。", grounding_value, v, "cue"
        )
        c2 = r.claim(
            "c2",
            "f1",
            "viewpoint",
            f"视点从低位升高约{rise:.1f}米并向下俯视{round(pitches[2]):.0f}度。",
            {"rise_m": round(rise, 3), "pitch_deg": pitches[2]},
            v,
            "transform",
        )
        c3 = r.claim("c3", "b1", "persistence", "物体世界位置保持不变。", True, v, "conclusion")
        r.sentence("cue", f"{name}等地标在三张图中仍能连续对应。", [c1])
        r.sentence(
            "transform",
            f"真正改变的是相机：它从低位升高约{rise:.1f}米，并向下俯视约{round(pitches[2]):.0f}度。",
            [c2],
        )
        r.sentence(
            "conclusion", "所以这是高处俯视造成的外观变化，不能解释成房间或家具整体下移。", [c3]
        )
        return r


class T8Planner(BasePlanner):
    """Occlusion, calibrated unknown, and evidence-conditioned revision."""

    def compile(self) -> dict[str, Any]:
        v = self.c.view_ids
        if len(v) != 6:
            raise PlanError("T8 requires six views")
        target = self.c.entity(self.c.selection["target_entity_id"])
        occluder = self.c.entity(self.c.selection["occluder_entity_id"])
        decisive = self.c.plan["occlusion_annotation"]["decisive_view"]
        if decisive != v[-1] or self.c.pixels(target, decisive) < MIN_PIXELS:
            raise PlanError("T8 decisive view fails compiler target visibility gate")
        occ_ref = self.local_ref(occluder, v[0])
        r1 = RoundBuilder(
            turn_id="t8-r01-occluded-unknown",
            new_view_ids=[v[0]],
            evidence_view_ids=[v[0]],
            capability={
                "primary": "CR",
                "supporting": ["SR"],
                "sense_nova_subtask": "epistemic boundary under occlusion",
            },
            semantic_signature="G_occluder->B_target_unknown->V_abstain",
            question_zh=f"有人说{occ_ref}后面可能还有一个{target.name}。只看第1个视角，现在能确认它存在或不存在吗？",
            answer_key={"target_status": "unknown"},
            diagnostics={"target_pixels": self.c.pixels(target, v[0])},
        )
        r1.node("g1", "G", ["view", "occluder"], "entity", occluder.source_id)
        r1.node(
            "b1",
            "B",
            ["prior_belief", "visibility_observation"],
            "belief_unknown",
            {"target": target.source_id, "status": "unknown"},
            ["g1"],
        )
        r1.node("v1", "V", ["belief", "visibility_threshold"], "unknown", "unknown", ["b1"])
        c11 = r1.claim(
            "c1",
            "g1",
            "visibility",
            f"第1个视角能看到{occ_ref}，但没有可靠看到{target.name}。",
            {"target_pixels": self.c.pixels(target, v[0])},
            [v[0]],
            "cue",
        )
        c12 = r1.claim(
            "c2",
            "v1",
            "epistemic",
            "未观察到目标不能推出目标不存在。",
            "unknown",
            [v[0]],
            "calibration",
        )
        r1.sentence("cue", f"第1个视角能看到{occ_ref}，但没有可靠看到{target.name}。", [c11])
        r1.sentence("calibration", "因此目前只能说无法确定；没看到不等于不存在。", [c12])

        middle = v[1:5]
        first_seen = self.c.well_visible(target, middle)
        reliable_middle = [view for view in middle if self.c.pixels(target, view) >= MIN_PIXELS]
        if first_seen is None:
            middle_status, middle_answer = "unknown", "仍然无法确认"
        else:
            target_local = self.local_ref(target, first_seen)
            persistence = (
                "，而且后续视角继续提供了同一目标的证据" if len(reliable_middle) > 1 else ""
            )
            middle_status, middle_answer = (
                "observed",
                f"已经可以确认，{target_local}在{self.view_ref(first_seen)}显露出来{persistence}",
            )
        r2 = RoundBuilder(
            turn_id="t8-r02-belief-revision",
            new_view_ids=middle,
            evidence_view_ids=[v[0], *middle],
            capability={
                "primary": "CR",
                "supporting": ["SR"],
                "sense_nova_subtask": "incremental belief revision",
            },
            semantic_signature="G_sequence->B_update->V_evidence_conditioned",
            question_zh="继续移动并看完第2到第5个视角。现在关于那个可能被挡住的目标，证据状态有没有改变？",
            answer_key={"target_status": middle_status, "first_reliable_view": first_seen},
        )
        r2.node(
            "g1",
            "G",
            ["view_sequence", "target"],
            "visibility_track",
            {view: self.c.pixels(target, view) for view in [v[0], *middle]},
        )
        r2.node(
            "b1",
            "B",
            ["prior_belief", "visibility_track"],
            "belief_scene_state",
            {"target": target.source_id, "status": middle_status},
            ["g1"],
        )
        r2.node(
            "v1",
            "V",
            ["belief", "instance_masks"],
            "boolean" if first_seen else "unknown",
            True if first_seen else "unknown",
            ["b1"],
        )
        c21 = r2.claim(
            "c1",
            "g1",
            "visibility_timeline",
            f"第2到第5个视角对目标的可靠可见状态为{middle_status}。",
            {"first_seen": first_seen},
            [v[0], *middle],
            "cue",
        )
        c22 = r2.claim(
            "c2",
            "b1",
            "belief_update",
            f"关于{target.name}的状态更新为{middle_status}。",
            middle_status,
            [v[0], *middle],
            "conclusion",
        )
        r2.sentence("cue", "我按顺序检查了新增视角，而不是沿用第1幅图的判断。", [c21])
        r2.sentence("conclusion", f"现在{middle_answer}。", [c22])

        target_ref = self.local_ref(target, decisive)
        r3 = RoundBuilder(
            turn_id="t8-r03-occlusion-explanation",
            new_view_ids=[decisive],
            evidence_view_ids=v,
            capability={
                "primary": "CR",
                "supporting": ["SR", "PT"],
                "sense_nova_subtask": "occlusion explanation and state consistency",
            },
            semantic_signature="G_reveal->F_path->B_revise->R_occluded_by->V",
            question_zh=f"最后一个视角清楚看到了{target_ref}。这说明它是后来才出现的，还是早先被视线遮住了？第1个视角的谨慎回答是否合理？",
            answer_key={
                "target_exists": True,
                "earlier_absence": "occluded",
                "initial_unknown_was_correct": True,
            },
        )
        r3.node(
            "g1", "G", ["view", "target"], "entity", {"target": target.source_id, "view": decisive}
        )
        r3.node("f1", "F", ["camera_pose_sequence"], "motion_path", "occlusion_to_reveal", ["g1"])
        r3.node(
            "b1",
            "B",
            ["prior_belief", "entity", "transform"],
            "belief_scene_state",
            {"target": target.source_id, "status": "observed"},
            ["g1", "f1"],
        )
        r3.node(
            "r1",
            "R",
            ["entity_pair", "view_sequence"],
            "occlusion_relation",
            {
                "target": target.source_id,
                "occluder": occluder.source_id,
                "occluded_views": [v[0]],
                "decisive_view": decisive,
            },
            ["b1"],
        )
        r3.node("v1", "V", ["occlusion", "quality_gate"], "boolean", True, ["r1"])
        c31 = r3.claim(
            "c1",
            "g1",
            "reveal",
            f"{target_ref}在最后一个视角可靠可见。",
            target.source_id,
            [decisive],
            "cue",
        )
        c32 = r3.claim(
            "c2",
            "f1",
            "viewpoint_change",
            "相机沿路径改变了与遮挡物的相对观察位置。",
            "occlusion_to_reveal",
            v,
            "transform",
        )
        c33 = r3.claim(
            "c3",
            "r1",
            "occlusion",
            f"目标早先被{occluder.name}遮住，并非后来生成。",
            "occluded",
            [v[0], decisive],
            "conclusion",
        )
        c34 = r3.claim(
            "c4",
            "v1",
            "calibration",
            "初始证据欠定时回答无法确定是正确的。",
            True,
            [v[0], decisive],
            "conclusion",
        )
        r3.sentence("cue", f"最后一个视角中，{target_ref}已经清楚显露。", [c31])
        r3.sentence(
            "transform", "移动改变了相机与遮挡物的相对位置，原先挡住目标的视线被打开。", [c32]
        )
        r3.sentence(
            "conclusion",
            f"所以{target.name}早先是被{occluder.name}遮住，并不是后来才出现；第1个视角回答无法确定是合理的。",
            [c33, c34],
        )
        return self.artifact([r1, r2, r3])


class T10Planner(BasePlanner):
    """Read relations in certified target views; does not claim prediction."""

    task_scope = "target_view_read_not_prediction"

    def compile(self) -> dict[str, Any]:
        v = self.c.view_ids
        anchors = self.c.plan.get("target_anchors", [])
        if len(v) != 3 or len(anchors) != 3 or not self.c.plan.get("held_out"):
            raise PlanError("T10 requires three held-out target anchor views")
        rounds: list[RoundBuilder] = []
        for index, (view, pair) in enumerate(zip(v, anchors, strict=True), start=1):
            origin = self.c.entity(pair["origin_entity_id"])
            facing = self.c.entity(pair["facing_entity_id"])
            if self.c.pixels(facing, view) < MIN_PIXELS:
                raise PlanError(f"T10 {view}: facing anchor below compiler pixel gate")
            candidates = []
            for target in self.c.named_visible(view):
                if target.source_id in {origin.source_id, facing.source_id}:
                    continue
                fact = self.c.object_frame_relation(origin, facing, target)
                if fact["hard_quadrant_ok"]:
                    candidates.append(
                        (fact["angle_margin_deg"], self.c.pixels(target, view), target, fact)
                    )
            if not candidates:
                raise PlanError(f"T10 {view}: no hard-margin target in object-anchored frame")
            _, _, target, fact = max(candidates)
            facing_ref = self.local_ref(facing, view)
            target_ref = self.local_ref(target, view)
            quadrant = QUADRANT_ZH[fact["quadrant"]]
            r = RoundBuilder(
                turn_id=f"t10-r{index:02d}-target-frame-read",
                new_view_ids=[view],
                evidence_view_ids=[view],
                capability={
                    "primary": "PT",
                    "supporting": ["SR"],
                    "sense_nova_subtask": "object-anchored target-view relation read",
                },
                semantic_signature="G_anchor+G_target->F_object_anchor->R_query->V",
                question_zh=f"{self.view_ref(view)}是在一{_near_classifier(origin)}{origin.name}附近选定站位，并面向{facing_ref}。以面向{facing.name}为正前方，{target_ref}在你的哪个方位？",
                answer_key={"quadrant": fact["quadrant"], "task_scope": self.task_scope},
                diagnostics={
                    "angle_margin_deg": fact["angle_margin_deg"],
                    "query_xy_m": fact["query_xy_m"],
                },
            )
            r.node("g1", "G", ["view", "facing_anchor"], "entity", facing.source_id)
            r.node("g2", "G", ["view", "target"], "entity", target.source_id)
            r.node(
                "f1",
                "F",
                ["origin_entity", "facing_entity"],
                "frame_object_anchored",
                {"origin": origin.source_id, "facing": facing.source_id},
                ["g1"],
            )
            r.node(
                "r1",
                "R",
                ["entity_pair", "object_anchored_frame"],
                "relation_quadrant",
                fact,
                ["g2", "f1"],
            )
            r.node("v1", "V", ["relation", "angle_extent_margin"], "boolean", True, ["r1"])
            c1 = r.claim(
                "c1",
                "g1",
                "anchor_grounding",
                f"镜头的正前方由{facing_ref}提供可见锚点。",
                facing.source_id,
                [view],
                "cue",
            )
            c2 = r.claim(
                "c2",
                "f1",
                "frame_transform",
                f"把站位到{facing.name}的方向定义为新的正前方。",
                {"origin": origin.source_id, "facing": facing.source_id},
                [view],
                "transform",
            )
            c3 = r.claim(
                "c3",
                "r1",
                "relation",
                f"{target.name}在该查询方向的{quadrant}。",
                fact,
                [view],
                "conclusion",
            )
            r.sentence("cue", f"先用{facing_ref}固定当前朝向，并找到{target_ref}。", [c1])
            r.sentence(
                "transform",
                f"把当前位置指向{facing.name}的方向当作正前方后，再重排左右和前后。",
                [c2],
            )
            r.sentence("conclusion", f"{target_ref}位于你的{quadrant}。", [c3])
            rounds.append(r)
        return self.artifact(rounds, task_scope=self.task_scope)


PLANNERS = {"T3": T3Planner, "T4": T4Planner, "T7": T7Planner, "T8": T8Planner, "T10": T10Planner}


def compile_bundle(
    bundle_root: Path,
    direction_balance: dict[str, int] | None = None,
) -> dict[str, Any]:
    try:
        trajectory_class = TrajectoryCatalog(bundle_root).trajectory_class
    except CatalogError as error:
        raise PlanError(str(error)) from error
    planner = PLANNERS.get(trajectory_class)
    if planner is None:
        raise PlanError(f"unsupported trajectory class: {trajectory_class}")
    return planner(bundle_root, direction_balance=direction_balance).compile()


def _near_classifier(entity: Entity) -> str:
    return (
        "张"
        if entity.label in {"desk", "breakfast_table", "coffee_table", "bed", "bench"}
        else "个"
    )

"""Role and cognitive-skill profiles for the Rs_int language subagents.

The profiles are language-generation policies, not geometry programs.  They
tell an answer narrator *how to verbalize* an already executed claim sheet;
they never grant the narrator permission to infer additional scene facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

QUESTION_EDITOR_ROLE: Final[str] = "question_editor"
ANSWER_NARRATOR_ROLE: Final[str] = "answer_narrator"
CRITIC_ROLE: Final[str] = "critic"

SENSENOVA_CAPABILITIES: Final[tuple[str, ...]] = ("MM", "SR", "MR", "PT", "CR")
TYPED_OPERATIONS: Final[tuple[str, ...]] = ("G", "F", "B", "M", "R", "P", "V")


@dataclass(frozen=True)
class NarrationSkillProfile:
    """A bounded verbalization policy for one human spatial strategy."""

    skill_id: str
    name_zh: str
    objective_zh: str
    required_moves_zh: tuple[str, ...]
    forbidden_moves_zh: tuple[str, ...]
    preferred_sentence_roles: tuple[str, ...]
    applicable_capabilities: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "name_zh": self.name_zh,
            "objective_zh": self.objective_zh,
            "required_moves_zh": list(self.required_moves_zh),
            "forbidden_moves_zh": list(self.forbidden_moves_zh),
            "preferred_sentence_roles": list(self.preferred_sentence_roles),
            "applicable_capabilities": list(self.applicable_capabilities),
        }

    def prompt_context_zh(self) -> str:
        required = "；".join(self.required_moves_zh)
        forbidden = "；".join(self.forbidden_moves_zh)
        return (
            f"本题采用‘{self.name_zh}’表达策略。目标：{self.objective_zh}\n"
            f"必须体现：{required}\n"
            f"禁止：{forbidden}"
        )


_SKILLS = {
    "visual_inventory": NarrationSkillProfile(
        skill_id="visual_inventory",
        name_zh="视觉清点",
        objective_zh="只列出当前画面中由实例证据支持的主要物体。",
        required_moves_zh=(
            "按 claim 给出的可见集合回答",
            "使用自然类别名并保持简洁",
        ),
        forbidden_moves_zh=(
            "凭房间类型补充画外物体",
            "把低置信的小物体写进清单",
        ),
        preferred_sentence_roles=("conclusion",),
        applicable_capabilities=("SR",),
    ),
    "co_visibility_then_metric": NarrationSkillProfile(
        skill_id="co_visibility_then_metric",
        name_zh="先共视、后度量",
        objective_zh="先确认两个物体是否由同一视角共同约束，再报告关系和允许的近似距离。",
        required_moves_zh=(
            "先给出是否同框及决定性视角",
            "再陈述 claim 许可的方向和近似数值",
        ),
        forbidden_moves_zh=(
            "把跨视图估计伪装成同框观察",
            "增加 claim 未许可的小数位或三维距离",
        ),
        preferred_sentence_roles=("evidence", "conclusion"),
        applicable_capabilities=("MM", "SR"),
    ),
    "route_replay": NarrationSkillProfile(
        skill_id="route_replay",
        name_zh="路线重放",
        objective_zh="用沿真实观察路线出现的关键地标，把未同框证据连接成一个结论。",
        required_moves_zh=(
            "只选 claim 中与结论直接相关的两个或三个路线检查点",
            "先说明物体分别在哪一段观察中被绑定，再给跨视图结论",
            "若 claim 允许，可说明两者未同框但可由共同路线对齐",
        ),
        forbidden_moves_zh=(
            "逐帧复述整条路线",
            "杜撰未列入 claim 的转弯、房间或地标",
            "使用相机轨迹对齐、坐标注册、canonical、DAG 等内部术语",
        ),
        preferred_sentence_roles=("evidence", "transform", "conclusion"),
        applicable_capabilities=("SR", "PT", "CR"),
    ),
    "landmark_hierarchy": NarrationSkillProfile(
        skill_id="landmark_hierarchy",
        name_zh="地标层级",
        objective_zh="先用功能区或稳定地标组织场景，再陈述局部物体关系。",
        required_moves_zh=(
            "仅在 claim 明确支持时区分厨房段、客厅段等区域",
            "以稳定地标作为关系锚点，避免无关物体清单",
            "从区域关系收束到题目要求的物体关系",
        ),
        forbidden_moves_zh=(
            "凭住宅常识补全卧室或未观察区域",
            "把同一视角中的二维左右偷换成全局关系",
            "写成固定模板式的完整场景描述",
        ),
        preferred_sentence_roles=("evidence", "conclusion"),
        applicable_capabilities=("SR", "CR"),
    ),
    "mental_simulation": NarrationSkillProfile(
        skill_id="mental_simulation",
        name_zh="视角心理模拟",
        objective_zh="先用已观察线索连接目标和视角锚点，再在指定站位与朝向下完成一次可复核的视角变换。",
        required_moves_zh=(
            "先引用至少一条实例专属视觉或记忆线索",
            "再明确采用题目给定的站位与面向作为新的正前方",
            "最后只重述 claim 许可且通过边界验证的相对方向",
        ),
        forbidden_moves_zh=(
            "沿用相机原始朝向回答 object-centric 问题",
            "暴露旋转矩阵、世界坐标或数值计算过程",
            "把角度阈值、物体包围范围或方向分界写进自然回答",
            "把心理模拟扩写成无法核验的长思维链",
        ),
        preferred_sentence_roles=("evidence", "transform", "conclusion"),
        applicable_capabilities=("SR", "PT"),
    ),
    "temporal_recall": NarrationSkillProfile(
        skill_id="temporal_recall",
        name_zh="时序检索",
        objective_zh="依据物体出现、消失和重现的观察序列回答最后可见或闭环持久性问题。",
        required_moves_zh=(
            "区分当前是否可见与历史是否见过",
            "仅引用 claim 指定的最后可见视角或重现视角",
            "闭环问题只比较首尾证据，不宣称未观测期间没有变化",
        ),
        forbidden_moves_zh=(
            "把最后一次看见误写成物体离开场景",
            "杜撰中间视角中的可见性",
            "用生活常识替代观测记录",
        ),
        preferred_sentence_roles=("conclusion", "evidence"),
        applicable_capabilities=("CR", "PT", "SR"),
    ),
    "calibration": NarrationSkillProfile(
        skill_id="calibration",
        name_zh="认知校准",
        objective_zh="严格区分从未观察到、证据不足与场景中不存在。",
        required_moves_zh=(
            "先陈述给定视图中的观察边界",
            "证据不足时明确回答无法确定",
            "置信措辞必须与 claim 的 epistemic_scope 一致",
        ),
        forbidden_moves_zh=(
            "把没有看到写成场景中不存在",
            "凭住宅类别猜测床、门或房间必然存在",
            "给 unknown 问题编造概率或置信数值",
        ),
        preferred_sentence_roles=("calibration",),
        applicable_capabilities=("CR", "SR", "MR"),
    ),
    "elimination": NarrationSkillProfile(
        skill_id="elimination",
        name_zh="证据排除",
        objective_zh="用 claim 支持的决定性证据排除冲突候选，并给出唯一可验证结论。",
        required_moves_zh=(
            "只排除问题中已经给出的候选或 claim 明确编码的反事实",
            "指出最小决定性证据，不枚举无关事实",
            "结论必须与 answer_key 一致",
        ),
        forbidden_moves_zh=(
            "自己增加候选项",
            "把相关性证据写成因果证明",
            "为增强说服力添加 claim 之外的数字或方向",
        ),
        preferred_sentence_roles=("evidence", "conclusion"),
        applicable_capabilities=("CR", "SR", "PT", "MM"),
    ),
}

NARRATION_SKILLS: Final = MappingProxyType(_SKILLS)


def get_narration_skill(skill_id: str) -> NarrationSkillProfile:
    """Return a frozen skill profile or fail before a provider call."""

    try:
        return NARRATION_SKILLS[skill_id]
    except KeyError as error:
        allowed = ", ".join(sorted(NARRATION_SKILLS))
        raise ValueError(
            f"unsupported narration skill {skill_id!r}; choose one of: {allowed}"
        ) from error

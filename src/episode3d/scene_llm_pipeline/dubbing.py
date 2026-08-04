"""Human-cognition dubbing layer over compiled scene dialogue truth.

Distribution contract (vs. the template surfaces):
- answer style follows difficulty: easy rounds answer directly; integration
  rounds narrate cue -> transform -> conclusion with a sampled human strategy;
- hedging language is tied to geometric margin; qualitative precedes numeric;
- the narrator may only verbalize claims; directions and numbers are validated
  against the claim sheet and fall back to the template surface on failure.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .lexicon import QUADRANT_ZH, RELATION_ZH

STRATEGIES = ("route_replay", "landmark_partition", "mental_simulation", "elimination", "memory_check")
STYLE_BY_TURN = {
    "r01-inventory": ("direct", "清点式，短句直接列举，不要展开推理。"),
    "r02-covis-relation": ("brief", "先一句给出同框视角，再一句给方向与距离；口吻自然，不用术语。"),
    "r03-temporal-recall": ("brief", "像回忆一样作答，可带一点检索口吻（如「回想一下」），最多两句。"),
    "r04-crossview-relation": ("trace", "线索→整合→结论三步：先指出两者各自出现在哪些画面，再说明如何把前后观察拼到一起（可提掉头），最后给方向。"),
    "r05-object-perspective": ("trace", "先回忆锚点物体的画面线索，再描述把自己放到该位置转向后的心理模拟，最后给方位。"),
    "r06-epistemic-loop": ("calibrated", "诚实校准：说清「画面里没见过」与「场景里没有」的区别；闭环判断给依据。"),
    "r07-layout-summary": ("summary", "像向朋友描述房间一样连贯地总结，不要罗列句式重复的短句。"),
}
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
FORBIDDEN = ("+X", "+Y", "entity_id", "scene_ir", "relation_oracle", "坐标系", "OBB", "claim")


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


class ClaudeBackend:
    def __init__(self, cache_dir: Path, model: str | None = None, timeout: float = 240.0) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.timeout = timeout

    def complete_json(self, prompt: str) -> dict[str, Any]:
        key = _hash({"prompt": prompt, "model": self.model})
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["response"]
        command = ["claude", "-p", "--output-format", "text"]
        if self.model:
            command += ["--model", self.model]
        result = subprocess.run(command, input=prompt, capture_output=True, text=True, timeout=self.timeout)
        if result.returncode != 0:
            raise RuntimeError(f"claude cli failed: {result.stderr[:300]}")
        text = result.stdout.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise RuntimeError(f"no JSON in response: {text[:200]}")
        response = json.loads(match.group(0))
        cache_file.write_text(
            json.dumps({"prompt_sha256": key, "response": response}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        return response


def _allowed_directions(round_: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    text = json.dumps(round_["claim_sheet"], ensure_ascii=False) + round_["question_zh"]
    for word in list(RELATION_ZH.values()) + list(QUADRANT_ZH.values()):
        if word in text:
            allowed.add(word)
    return allowed


def _validate(answer: str, round_: dict[str, Any]) -> str | None:
    for token in FORBIDDEN:
        if token in answer:
            return f"forbidden token {token!r}"
    allowed = _allowed_directions(round_)
    for word in list(QUADRANT_ZH.values()) + list(RELATION_ZH.values()):
        if word in answer and word not in allowed:
            return f"direction {word!r} not licensed by claims"
    licensed_numbers = set(NUM_RE.findall(json.dumps(round_["claim_sheet"], ensure_ascii=False) + round_["question_zh"] + round_["answer_zh"]))
    for number in NUM_RE.findall(answer):
        if number not in licensed_numbers and not (number.isdigit() and int(number) <= 11):
            return f"number {number!r} not licensed by claims"
    return None


def dub_round(round_: dict[str, Any], backend: ClaudeBackend, strategy: str) -> dict[str, Any]:
    style_key, style_note = STYLE_BY_TURN.get(round_["turn_id"], ("brief", "自然口吻，两句以内。"))
    question_prompt = (
        "你是空间问答的出题编辑。把下面的问题蓝图改写成一句自然、口语化的中文问题。"
        "必须保留全部语义要素（提到的物体、视角编号、方位参照的含义），不得暗示答案，不得加入新事实。"
        f"只输出JSON：{{\"question_zh\": \"...\"}}\n问题蓝图：{round_['question_zh']}"
    )
    question = backend.complete_json(question_prompt).get("question_zh", round_["question_zh"])
    claims = [
        {"role": c.get("reasoning_role", "fact"), "statement": c["statement_zh"]}
        for c in round_["claim_sheet"]["claims"]
    ]
    answer_prompt = (
        "你是回答叙述者。只能使用下面的事实清单组织回答，禁止引入任何清单之外的空间断言、方向或数字；"
        "方向和数值必须与清单逐字一致。回答要像人一样自然。"
        f"风格要求：{style_note}"
        f"叙述策略参考：{strategy}。"
        f"只输出JSON：{{\"answer_zh\": \"...\"}}\n"
        f"问题：{question}\n事实清单：{json.dumps(claims, ensure_ascii=False)}"
    )
    dubbed = round_.copy()
    dubbed["question_zh_template"] = round_["question_zh"]
    dubbed["answer_zh_template"] = round_["answer_zh"]
    dubbed["question_zh"] = question
    for attempt in range(2):
        answer = backend.complete_json(answer_prompt).get("answer_zh", "")
        error = _validate(answer, round_)
        if error is None and answer:
            dubbed["answer_zh"] = answer
            dubbed["dubbing"] = {"strategy": strategy, "style": style_key, "attempts": attempt + 1, "fallback": False}
            return dubbed
        answer_prompt += f"\n上一次输出被拒绝（{error}），请修正后重新只输出JSON。"
    dubbed["answer_zh"] = round_["answer_zh"]
    dubbed["dubbing"] = {"strategy": strategy, "style": style_key, "attempts": 2, "fallback": True, "error": error}
    return dubbed


def dub_scene(artifact_path: Path, output_path: Path, cache_dir: Path, model: str | None = None) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    backend = ClaudeBackend(cache_dir, model=model)
    seed = int(_hash(artifact["scene_id"]), 16)
    rounds = []
    for index, round_ in enumerate(artifact["rounds"]):
        strategy = STRATEGIES[(seed + index) % len(STRATEGIES)]
        rounds.append(dub_round(round_, backend, strategy))
    artifact["rounds"] = rounds
    artifact["dubbing_provenance"] = {"backend": "claude-cli", "model": model or "session-default"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    return artifact

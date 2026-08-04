# ruff: noqa: E501, RUF001
"""Build one landing page for static and intervention human review desks."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from omnigibson_episode.pair_review import build_intervention_review
from omnigibson_episode.review import build_sweep_review

_PRODUCTION_INTERVENTION_SWEEPS = {
    "relation-flip-m2-room-safe-v2",
    "model-swap-m2-kinematic-v5",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def build_review_portal(
    *, static_plan: Path, intervention_plans: list[Path], output_path: Path
) -> Path:
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    static_page = build_sweep_review(static_plan)
    static_data = _read(static_page.parent / "review_data.json")
    desks = [
        {
            "kind": "static",
            "stage": "production",
            "title": "静态多视图基线",
            "subtitle": "先检查场景、全模态证据与 8 类可执行推理任务",
            "sweep_id": static_data["sweep_id"],
            "count": len(static_data["records"]),
            "actionable_count": len(static_data["records"]),
            "status_counts": static_data["status_counts"],
            "link": os.path.relpath(static_page, output.parent),
        }
    ]
    for plan in intervention_plans:
        page = build_intervention_review(plan)
        data = _read(page.parent / "review_data.json")
        kind = str(data.get("intervention_kind") or "intervention")
        sweep_id = str(data["sweep_id"])
        stage = (
            "production"
            if sweep_id in _PRODUCTION_INTERVENTION_SWEEPS
            else "historical_audit"
        )
        desks.append(
            {
                "kind": kind,
                "stage": stage,
                "title": "关系翻转" if kind == "relation_flip" else "同类模型替换",
                "subtitle": (
                    "确认只有一个对象位置和空间答案改变"
                    if kind == "relation_flip"
                    else "确认外观改变，但相机、位置与空间答案保持"
                ),
                "sweep_id": sweep_id,
                "count": len(data["records"]),
                "actionable_count": int(data.get("actionable_count", 0)),
                "status_counts": data["status_counts"],
                "link": os.path.relpath(page, output.parent),
            }
        )
    desks.sort(
        key=lambda item: (
            0
            if item["kind"] == "static"
            else 1
            if item["stage"] == "production"
            else 2,
            0 if item["kind"] == "relation_flip" else 1,
            str(item["sweep_id"]),
        )
    )
    output.write_text(_page(desks), encoding="utf-8")
    return output


def _page(desks: list[dict[str, Any]]) -> str:
    cards = []
    for desk in desks:
        stage = str(desk.get("stage", "production"))
        stage_label = "PRODUCTION" if stage == "production" else "HISTORICAL AUDIT"
        actionable_count = int(desk.get("actionable_count", desk["count"]))
        retained_audit_count = int(desk["count"]) - actionable_count
        counts = "".join(
            f"<span><b>{int(value)}</b>{html.escape(str(key))}</span>"
            for key, value in sorted(desk["status_counts"].items())
        )
        cards.append(
            f"""<article class="{html.escape(stage)}"><div class="tag">{stage_label} · {html.escape(desk['kind'])}</div>
<h2>{html.escape(desk['title'])}</h2><p>{html.escape(desk['subtitle'])}</p>
<code>{html.escape(desk['sweep_id'])}</code><div class="counts"><span><b>{desk['count']}</b>计划记录</span><span><b>{actionable_count}</b>待人工队列</span><span><b>{retained_audit_count}</b>自动审计记录</span>{counts}</div>
<a href="{html.escape(desk['link'])}">{'进入正式审阅台' if stage == 'production' else '查看失败模式'} →</a></article>"""
        )
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Episode3D Review Hub</title>
<style>:root{{--bg:#07111f;--panel:#0e1c2e;--line:#29415c;--text:#eef7ff;--muted:#9cb0c6;--cyan:#59e1ff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0,#12304a,#07111f 42%);color:var(--text);font:15px/1.55 Inter,system-ui}}main{{max-width:1250px;margin:auto;padding:52px 28px}}h1{{font-size:clamp(38px,6vw,72px);letter-spacing:-.05em;margin:0}}.lead{{color:var(--muted);max-width:850px;font-size:17px}}.steps{{display:flex;gap:10px;flex-wrap:wrap;margin:28px 0}}.steps span,.tag{{border:1px solid var(--line);border-radius:999px;padding:5px 10px;color:var(--cyan)}}section{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}}article{{background:rgba(14,28,46,.94);border:1px solid var(--line);border-radius:18px;padding:22px}}article.historical_audit{{opacity:.68}}article.historical_audit .tag{{color:#9cb0c6}}h2{{font-size:24px;margin:14px 0 4px}}p,code{{color:var(--muted)}}code{{display:block;overflow-wrap:anywhere}}.counts{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}}.counts span{{background:#091726;border-radius:9px;padding:8px 10px}}.counts b{{display:block;color:var(--cyan);font-size:21px}}a{{display:inline-block;color:#07111f;background:var(--cyan);padding:9px 14px;border-radius:9px;text-decoration:none;font-weight:700}}</style>
<main><h1>Episode3D Review Hub</h1><p class="lead">以 acquisition 和 minimal pair 为单位审阅，不以派生 QA 数量制造规模。自动证书负责几何与因果约束，具名 reviewer 负责画面、语义和训练价值。</p>
<div class="steps"><span>1 · 静态 base</span><span>2 · 全模态证据</span><span>3 · 自然语言任务</span><span>4 · 成对干预</span><span>5 · 导出审阅 JSON</span></div>
<section>{''.join(cards)}</section></main></html>"""

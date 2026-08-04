# ruff: noqa: E501, RUF001
"""Build a side-by-side human review desk for intervention minimal pairs."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from omnigibson_episode.intervention_sweep import refresh_intervention_sweep_plan
from omnigibson_episode.io import write_json_atomic


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _relative(path: Path, review_root: Path) -> str:
    return os.path.relpath(path.resolve(), review_root.resolve())


def _record(job: dict[str, Any], review_root: Path) -> dict[str, Any]:
    base = Path(job["base_bundle"])
    variant = Path(job["bundle"])
    pair = _optional_json(variant / "minimal_pair.json")
    failure = _optional_json(variant / "failure_report.json")
    certification_failure = _optional_json(variant / "minimal_pair_failure.json")
    quality = _optional_json(variant / "quality_report.json")
    proposal_plan = _optional_json(Path(job["proposal_plan"]))
    proposal = next(
        (
            item
            for item in proposal_plan.get("proposals", [])
            if item.get("proposal_id") == job["proposal_id"]
        ),
        {},
    )
    pair_evidence = pair.get("evidence", {})
    base_target_views = set(pair_evidence.get("base", {}).get("target_view_ids", []))
    variant_target_views = set(pair_evidence.get("variant", {}).get("target_view_ids", []))
    matched = sorted(base_target_views & variant_target_views)
    if not matched:
        matched = list(proposal.get("target_visible_view_ids_before", []))
    matched = matched[:3]
    view_pairs = []
    for view_id in matched:
        base_image = base / "preview" / f"{view_id}.png"
        variant_image = variant / "preview" / f"{view_id}.png"
        view_pairs.append(
            {
                "view_id": view_id,
                "base": _relative(base_image, review_root) if base_image.is_file() else None,
                "variant": (
                    _relative(variant_image, review_root)
                    if variant_image.is_file()
                    else None
                ),
            }
        )
    episodes = pair.get("episodes", {})
    return {
        "job_id": job["job_id"],
        "scene_model": job["scene_model"],
        "status": job["status"],
        "review_actionable": job["status"] in {"certified", "needs_review"},
        "status_detail": job.get("status_detail"),
        "intervention_type": job.get(
            "intervention_type", proposal_plan.get("intervention_type")
        ),
        "proposal_id": job["proposal_id"],
        "target": proposal.get("target_source_entity_id"),
        "anchor": proposal.get("anchor_source_entity_id"),
        "source_model": proposal.get("source_model"),
        "replacement_model": proposal.get("replacement_model"),
        "execution_mode": proposal.get("execution_mode"),
        "settle_steps": proposal.get("settle_steps"),
        "baseline_overlap_count": proposal.get("baseline_overlap_count"),
        "minimum_nonoverlap_clearance_m": proposal.get(
            "minimum_nonoverlap_clearance_m"
        ),
        "learning_signal": pair.get("learning_signal"),
        "answer_base": episodes.get("base", {}).get(
            "answer", proposal.get("relation_before")
        ),
        "answer_variant": episodes.get("variant", {}).get(
            "answer", proposal.get("relation_after")
        ),
        "view_pairs": view_pairs,
        "checks": pair.get(
            "checks", failure.get("checks", certification_failure.get("checks", []))
        ),
        "failure_reason": (
            failure.get("error", {}).get("message")
            or certification_failure.get("error", {}).get("message")
            or job.get("status_detail")
        ),
        "quality": {
            "integrity_status": quality.get("integrity_status"),
            "visual_status": quality.get("visual_status"),
            "warnings": quality.get("warnings", []),
        },
        "base_preview": (
            _relative(base / "preview.html", review_root)
            if (base / "preview.html").is_file()
            else None
        ),
        "variant_preview": (
            _relative(variant / "preview.html", review_root)
            if (variant / "preview.html").is_file()
            else None
        ),
    }


def build_intervention_review(
    plan_path: Path, output_directory: Path | None = None
) -> Path:
    plan = refresh_intervention_sweep_plan(plan_path.resolve())
    root = Path(plan["output_root"])
    review_root = (output_directory or root / "review").resolve()
    review_root.mkdir(parents=True, exist_ok=True)
    records = [_record(job, review_root) for job in plan["jobs"]]
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    payload = {
        "schema_version": "omnigibson_intervention_review.v1",
        "sweep_id": plan["sweep_id"],
        "intervention_kind": plan.get("intervention_kind"),
        "status_counts": dict(sorted(counts.items())),
        "actionable_count": sum(record["review_actionable"] for record in records),
        "records": records,
    }
    write_json_atomic(review_root / "review_data.json", payload)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    output = review_root / "index.html"
    output.write_text(_page(data), encoding="utf-8")
    return output


def _page(data: str) -> str:
    title = html.escape("Episode3D · Minimal Pair 审阅台")
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{--bg:#07111f;--panel:#0e1c2e;--line:#29415c;--text:#eef7ff;--muted:#9cb0c6;--cyan:#59e1ff;--green:#7ff0ae;--amber:#ffd477;--red:#ff8a96}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#07111f,#0a1726 55%,#07111f);color:var(--text);font:14px/1.5 Inter,system-ui}}main{{max-width:1500px;margin:auto;padding:32px}}h1{{font-size:clamp(28px,4vw,52px);margin:0}}.lead{{color:var(--muted);max-width:900px}}.top,.toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}}.stat,article{{background:var(--panel);border:1px solid var(--line);border-radius:14px}}.stat{{padding:12px 16px}}.stat b{{display:block;color:var(--cyan);font-size:24px}}.stat.remaining b{{color:var(--amber)}}input,select,button,textarea,.file{{background:#0b1a2b;color:var(--text);border:1px solid var(--line);border-radius:9px;padding:9px 12px}}.file,button{{cursor:pointer}}.reviewer{{min-width:220px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}}article{{overflow:hidden}}article.audit-only{{opacity:.72}}.head,.body{{padding:14px 17px}}.head{{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line)}}h2{{font-size:18px;margin:0}}.passed,.certified{{color:var(--green)}}.needs_review,.pending,.running{{color:var(--amber)}}.failed{{color:var(--red)}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#050b13}}.pair figure{{margin:0}}.pair img{{display:block;width:100%;height:220px;object-fit:cover}}figcaption{{padding:5px 9px;color:var(--muted)}}.signal{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}}.pill{{border:1px solid var(--line);border-radius:999px;padding:3px 9px}}.answer{{font-size:17px}}.arrow{{color:var(--cyan);padding:0 8px}}.failure,.gate-note{{color:var(--red);background:#28131b;border-radius:8px;padding:8px 10px}}details{{margin:10px 0}}.checks{{font:12px ui-monospace,monospace;color:var(--muted)}}.decision{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}}.decision .active{{outline:2px solid var(--cyan)}}textarea{{width:100%;min-height:55px;margin-top:8px}}a{{color:var(--cyan)}}.empty{{padding:35px;color:var(--muted)}}@media(max-width:650px){{main{{padding:16px}}.grid{{grid-template-columns:1fr}}.pair img{{height:150px}}}}
</style><main><h1>Minimal Pair 审阅台</h1><p class="lead">默认只显示通过自动门禁、需要具名人工决定的候选。左右严格对应同一相机位姿：关系翻转只应改变一个物体的位置和答案；模型替换应改变外观但保持空间答案。自动失败记录保留为审计证据，不能被人工提升。</p>
<div class="top" id="stats"></div><div class="toolbar"><input id="search" placeholder="搜索场景或目标"><select id="status"><option value="actionable">候选审核队列（默认）</option><option value="unreviewed">仅未审候选</option><option value="audit">自动排除记录</option><option value="all">全部记录</option></select><button id="next">下一个未审候选</button><input class="reviewer" id="reviewer" placeholder="审阅人（必填）"><label class="file">导入审阅 JSON<input id="import" type="file" accept="application/json" hidden></label><button id="export">导出审阅 JSON</button></div><section class="grid" id="grid"></section></main>
<script id="data" type="application/json">{data}</script><script>
const data=JSON.parse(document.querySelector('#data').textContent),key='episode3d-pair-review:'+data.sweep_id;
let decisions=JSON.parse(localStorage.getItem(key)||'{{}}');const reviewer=document.querySelector('#reviewer'),reviewerKey=key+':reviewer';reviewer.value=localStorage.getItem(reviewerKey)||'';reviewer.onchange=()=>localStorage.setItem(reviewerKey,reviewer.value.trim());
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const status=document.querySelector('#status');
Object.keys(data.status_counts).forEach(x=>status.add(new Option('状态 · '+x,'status:'+x)));
const decided=id=>['accept','review','reject'].includes(decisions[id]?.decision);
function matchesQueue(r,sf){{
  if(sf==='all')return true;
  if(sf==='actionable')return r.review_actionable;
  if(sf==='unreviewed')return r.review_actionable&&!decided(r.job_id);
  if(sf==='audit')return !r.review_actionable;
  if(sf.startsWith('status:'))return r.status===sf.slice(7);
  return false;
}}
function visibleRecords(){{
  let q=document.querySelector('#search').value.toLowerCase(),sf=status.value;
  return data.records.filter(r=>matchesQueue(r,sf)&&(`${{r.scene_model}} ${{r.target}}`).toLowerCase().includes(q));
}}
function render(){{
  let actionable=data.records.filter(r=>r.review_actionable),reviewed=actionable.filter(r=>decided(r.job_id)).length,
      remaining=actionable.length-reviewed,visible=visibleRecords();
  document.querySelector('#stats').innerHTML=`<div class="stat"><b>${{data.records.length}}</b>全部计划 pairs</div><div class="stat"><b>${{actionable.length}}</b>候选审核队列</div><div class="stat"><b>${{reviewed}}</b>候选已审</div><div class="stat remaining"><b>${{remaining}}</b>候选待审</div>`+Object.entries(data.status_counts).map(([k,v])=>`<div class="stat"><b>${{v}}</b>${{esc(k)}}</div>`).join('');
  document.querySelector('#grid').innerHTML=visible.map(r=>{{
    let d=decisions[r.job_id]||{{decision:'unreviewed',note:''}},views=r.view_pairs.length?r.view_pairs.map(v=>`<div class="pair"><figure>${{v.base?`<img loading="lazy" src="${{esc(v.base)}}">`:'<div class="empty">base 待生成</div>'}}<figcaption>Base · ${{esc(v.view_id)}}</figcaption></figure><figure>${{v.variant?`<img loading="lazy" src="${{esc(v.variant)}}">`:'<div class="empty">variant 待生成</div>'}}<figcaption>Variant · ${{esc(v.view_id)}}</figcaption></figure></div>`).join(''):'<div class="empty">等待匹配视角</div>';
    let failed=(r.checks||[]).filter(x=>!x.passed),policy=r.execution_mode?`<span class="pill">${{esc(r.execution_mode)}} · settle=${{esc(r.settle_steps)}}</span>`:'',clearance=r.minimum_nonoverlap_clearance_m!=null?`<span class="pill">overlap=${{esc(r.baseline_overlap_count)}} · clearance=${{esc(r.minimum_nonoverlap_clearance_m)}}m</span>`:'',reason=r.failure_reason?`<p class="failure">${{r.review_actionable?'复核原因':'自动排除原因'}}：${{esc(r.failure_reason)}}</p>`:'',controls=r.review_actionable?`<div class="decision">${{['accept','review','reject'].map(x=>`<button data-id="${{r.job_id}}" data-d="${{x}}" class="${{d.decision===x?'active':''}}">${{{{accept:'通过',review:'复核',reject:'拒绝'}}[x]}}</button>`).join('')}}</div><textarea data-note="${{r.job_id}}" placeholder="审阅备注">${{esc(d.note)}}</textarea>`:'<p class="gate-note">该记录已被自动门禁排除，仅供失败模式审计；网页不提供人工提升按钮。</p>';
    return `<article class="${{r.review_actionable?'':'audit-only'}}" data-job-id="${{esc(r.job_id)}}"><div class="head"><div><h2>${{esc(r.scene_model)}}</h2><span>${{esc(r.target)}} · ${{esc(r.proposal_id)}}</span></div><strong class="${{esc(r.status)}}">${{esc(r.status)}}</strong></div>${{views}}<div class="body"><div class="signal"><span class="pill">${{esc(r.intervention_type)}}</span><span class="pill">${{esc(r.learning_signal||'无训练证书')}}</span>${{policy}}${{clearance}}<span class="pill">visual=${{esc(r.quality.visual_status)}}</span></div>${{reason}}<div class="answer">GT：${{esc(r.answer_base)}}<span class="arrow">→</span>${{esc(r.answer_variant)}}</div>${{r.source_model?`<p>资产：${{esc(r.source_model)}} → ${{esc(r.replacement_model)}}</p>`:''}}<details><summary>${{r.checks.length}} 项 certificate checks · 失败 ${{failed.length}}</summary><div class="checks">${{r.checks.map(x=>`${{x.passed?'✓':'✗'}} ${{esc(x.name)}} · ${{esc(x.measured_value)}} / ${{esc(x.threshold)}}`).join('<br>')}}</div></details><p>${{r.base_preview?`<a target="_blank" href="${{esc(r.base_preview)}}">Base 全模态</a>`:''}} ${{r.variant_preview?` · <a target="_blank" href="${{esc(r.variant_preview)}}">Variant 全模态</a>`:''}}</p>${{controls}}</div></article>`;
  }}).join('');
  document.querySelectorAll('[data-d]').forEach(b=>b.onclick=()=>{{let id=b.dataset.id;decisions[id]={{...(decisions[id]||{{}}),decision:b.dataset.d,updated_at_utc:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(decisions));render()}});
  document.querySelectorAll('[data-note]').forEach(t=>t.onchange=()=>{{let id=t.dataset.note;decisions[id]={{...(decisions[id]||{{decision:'unreviewed'}}),note:t.value,updated_at_utc:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(decisions));render()}});
}}
document.querySelector('#next').onclick=()=>{{status.value='unreviewed';render();requestAnimationFrame(()=>document.querySelector('#grid article')?.scrollIntoView({{behavior:'smooth',block:'start'}}))}};
document.querySelector('#search').oninput=render;status.onchange=render;
document.querySelector('#import').onchange=async e=>{{let f=e.target.files[0];if(!f)return;try{{let p=JSON.parse(await f.text());if(p.schema_version!=='omnigibson_intervention_human_review.v1'||p.sweep_id!==data.sweep_id)throw new Error('schema 或 sweep_id 不匹配');decisions=p.decisions||{{}};reviewer.value=p.reviewer||'';localStorage.setItem(key,JSON.stringify(decisions));localStorage.setItem(reviewerKey,reviewer.value.trim());render()}}catch(err){{alert('导入失败：'+err.message)}}}};
document.querySelector('#export').onclick=()=>{{let id=reviewer.value.trim();if(!id){{alert('请先填写审阅人');return}}let remaining=data.records.filter(r=>r.review_actionable&&!decided(r.job_id)).length;if(remaining&&!confirm(`仍有 ${{remaining}} 个候选未审。是否导出检查点 JSON？`))return;let blob=new Blob([JSON.stringify({{schema_version:'omnigibson_intervention_human_review.v1',sweep_id:data.sweep_id,reviewer:id,generated_at_utc:new Date().toISOString(),decisions}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=data.sweep_id+'.human-review.json';a.click();URL.revokeObjectURL(a.href)}};render();
</script></html>"""

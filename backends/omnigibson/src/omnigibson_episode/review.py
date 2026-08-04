# ruff: noqa: E501, RUF001
"""Build a dependency-free human review desk for a scene sweep."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from omnigibson_episode.io import write_json_atomic
from omnigibson_episode.sweep import refresh_sweep_plan


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _review_record(job: dict[str, Any], review_root: Path) -> dict[str, Any]:
    bundle = Path(job["bundle"])
    quality = _optional_json(bundle / "quality_report.json")
    audit = _optional_json(bundle / "reasoning_audit.json")
    episode = _optional_json(bundle / "spatial_episode.json")
    snapshot = _optional_json(bundle / "scene_snapshot.json")
    task_manifest = _optional_json(bundle / "reasoning_tasks.json")
    metrics = quality.get("views", [])
    selected: list[dict[str, str]] = []
    if metrics:
        usable = metrics[:-1] if len(metrics) > 1 else metrics
        boundaries = [
            usable[: max(1, len(usable) // 3)],
            usable[len(usable) // 3 : max(len(usable) // 3 + 1, 2 * len(usable) // 3)],
            usable[max(2 * len(usable) // 3, 0) :],
        ]
        for group in boundaries:
            if not group:
                continue
            item = max(group, key=lambda value: int(value["visible_instance_count"]))
            preview = bundle / item["preview"]
            selected.append(
                {
                    "view_id": item["view_id"],
                    "preview": str(preview.relative_to(review_root.parent)),
                }
            )
    return {
        "job_id": job["job_id"],
        "scene_model": job["scene_model"],
        "classification": job["classification"],
        "status": job["status"],
        "status_detail": job.get("status_detail"),
        "bundle": str(bundle),
        "preview_html": (
            str((bundle / "preview.html").relative_to(review_root.parent))
            if (bundle / "preview.html").is_file()
            else None
        ),
        "selected_views": selected,
        "quality": {
            "visual_status": quality.get("visual_status"),
            "warnings": quality.get("warnings", []),
            **quality.get("summary", {}),
            "loop_closure": quality.get("loop_closure"),
        },
        "evidence": audit.get("evidence_summary", {}),
        "capabilities": audit.get("capability_matrix", {}),
        "reasoning_status": audit.get("reasoning_status"),
        "evidence_eligibility": audit.get("evidence_eligibility"),
        "gaps": audit.get("gaps", []),
        "reasoning_tasks": [
            {
                "task_type": task.get("task_type"),
                "capabilities": task.get("capabilities", []),
                "question_zh": task.get("question_zh"),
                "surface_answer_zh": task.get("surface_answer_zh"),
                "status": task.get("status"),
                "certificate_result": task.get("certificate", {}).get("result"),
                "program": "→".join(
                    node.get("operation", "?")
                    for node in task.get("operation_graph", {}).get("nodes", [])
                ),
            }
            for task in task_manifest.get("tasks", [])
        ],
        "reasoning_task_coverage": task_manifest.get("coverage_status"),
        "episode_id": episode.get("episode_id"),
        "query_count": len(episode.get("queries", [])),
        "entity_count": len(snapshot.get("entities", [])),
    }


def build_sweep_review(plan_path: Path, output_directory: Path | None = None) -> Path:
    plan_path = plan_path.resolve()
    plan = refresh_sweep_plan(plan_path)
    output_root = Path(plan["output_root"])
    review_root = (output_directory or output_root / "review").resolve()
    review_root.mkdir(parents=True, exist_ok=True)
    records = [_review_record(job, review_root) for job in plan["jobs"]]
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    payload = {
        "schema_version": "omnigibson_sweep_review.v1",
        "sweep_id": plan["sweep_id"],
        "status_counts": dict(sorted(counts.items())),
        "independent_acquisition_count": sum(
            record["status"] in {"passed", "needs_review"} for record in records
        ),
        "records": records,
    }
    write_json_atomic(review_root / "review_data.json", payload)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = _page(data)
    output = review_root / "index.html"
    output.write_text(page, encoding="utf-8")
    return output


def _page(data: str) -> str:
    title = html.escape("Episode3D · OmniGibson 场景审阅台")
    return f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#07111f;--panel:#0e1c2e;--line:#29415c;--text:#eef7ff;--muted:#9cb0c6;
--cyan:#59e1ff;--green:#7ff0ae;--amber:#ffd477;--red:#ff8a96}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07111f,#0a1726 55%,#07111f);
color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui}} main{{max-width:1500px;margin:auto;padding:34px}}
h1{{font-size:clamp(28px,4vw,54px);margin:0 0 6px;letter-spacing:-.04em}} .lead{{color:var(--muted);max-width:850px}}
.top{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}} .stat{{background:#0c1c2d;border:1px solid var(--line);
border-radius:14px;padding:14px 18px;min-width:130px}} .stat b{{display:block;font-size:25px;color:var(--cyan)}} .stat.remaining b{{color:var(--amber)}}
.toolbar{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:20px 0}} input,select,button,textarea,.file{{
background:#0b1a2b;color:var(--text);border:1px solid var(--line);border-radius:9px;padding:9px 12px}} button{{cursor:pointer}}
.file{{cursor:pointer}} .reviewer{{min-width:220px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:18px}} article{{background:rgba(14,28,46,.92);
border:1px solid var(--line);border-radius:16px;overflow:hidden}} .head{{padding:16px 18px;display:flex;justify-content:space-between;
gap:12px}} h2{{font-size:18px;margin:0}} .pill{{border:1px solid var(--line);border-radius:999px;padding:3px 9px;color:var(--muted)}}
.passed{{color:var(--green)}} .needs_review,.running{{color:var(--amber)}} .failed,.incomplete{{color:var(--red)}}
.views{{display:grid;grid-template-columns:repeat(3,1fr);background:#050b13}} .views img{{width:100%;height:145px;object-fit:cover;
object-position:left top;border-right:1px solid #14263a}} .body{{padding:14px 18px}} .metrics{{display:grid;grid-template-columns:repeat(3,1fr);
gap:8px;margin-bottom:12px}} .metric{{background:#091726;border-radius:9px;padding:8px}} .metric small{{display:block;color:var(--muted)}}
.caps{{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}} .cap{{font:12px ui-monospace,monospace;border:1px solid var(--line);
border-radius:7px;padding:3px 6px}} .cap.yes{{border-color:#2c8c68;color:var(--green)}} .cap.ready{{border-color:#946f2b;color:var(--amber)}}
.decision{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:12px}} .decision button.active{{outline:2px solid var(--cyan)}}
textarea{{width:100%;min-height:55px;margin-top:8px;resize:vertical}} a{{color:var(--cyan)}} .empty{{color:var(--muted)}}
.tasks{{margin:12px 0;border-top:1px solid var(--line);padding-top:10px}} .tasks summary{{cursor:pointer;color:var(--cyan)}}
.task{{margin-top:9px;padding:9px 10px;background:#091726;border-radius:9px}} .task q{{display:block;color:var(--text)}}
.task .answer{{color:var(--green);margin-top:3px}} .task small{{color:var(--muted);font-family:ui-monospace,monospace}}
@media(max-width:600px){{main{{padding:18px}}.grid{{grid-template-columns:1fr}}.views img{{height:105px}}}}
</style>
<main><h1>Episode3D 场景审阅台</h1>
<p class="lead">统计单位是独立渲染的 scene/trajectory，而不是派生 QA 条数。每张卡同时展示画质、
证据覆盖和当前训练问题覆盖；“证据可支持”不等于“已经采样为训练题”。人工决定保存在浏览器本地。</p>
<div class="top" id="stats"></div>
<div class="toolbar"><input id="search" placeholder="搜索场景"><select id="status"><option value="">全部状态</option></select>
<button id="next">下一个未审场景</button><input class="reviewer" id="reviewer" placeholder="审阅人（必填）"><label class="file">导入审阅 JSON
<input id="import" type="file" accept="application/json" hidden></label><button id="export">导出人工审阅 JSON</button></div>
<section class="grid" id="grid"></section></main>
<script id="data" type="application/json">{data}</script>
<script>
const data=JSON.parse(document.querySelector('#data').textContent), key='episode3d-review:'+data.sweep_id;
let decisions=JSON.parse(localStorage.getItem(key)||'{{}}');
const reviewer=document.querySelector('#reviewer'), reviewerKey=key+':reviewer';
reviewer.value=localStorage.getItem(reviewerKey)||'';
reviewer.onchange=()=>localStorage.setItem(reviewerKey,reviewer.value.trim());
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const status=document.querySelector('#status'); Object.keys(data.status_counts).forEach(x=>status.add(new Option(x,x)));
function metric(label,value){{return `<div class="metric"><small>${{label}}</small>${{value??'—'}}</div>`}}
function render(){{let q=document.querySelector('#search').value.toLowerCase(),sf=status.value,
reviewed=data.records.filter(r=>['accept','review','reject'].includes(decisions[r.job_id]?.decision)).length;
document.querySelector('#stats').innerHTML=`<div class="stat"><b>${{data.records.length}}</b>目标场景</div>
<div class="stat"><b>${{data.independent_acquisition_count}}</b>已渲染 bundle</div><div class="stat"><b>${{reviewed}}</b>已人工审阅</div><div class="stat remaining"><b>${{data.records.length-reviewed}}</b>待人工审阅</div>`+
Object.entries(data.status_counts).map(([k,v])=>`<div class="stat"><b>${{v}}</b>${{esc(k)}}</div>`).join('');
document.querySelector('#grid').innerHTML=data.records.filter(r=>(!sf||r.status===sf)&&r.scene_model.toLowerCase().includes(q)).map(r=>{{
let d=decisions[r.job_id]||{{decision:'unreviewed',note:''}}, views=r.selected_views.length?r.selected_views.map(v=>
`<img loading="lazy" src="../${{esc(v.preview)}}" title="${{esc(v.view_id)}}">`).join(''):'<div class="empty">尚无预览</div>';
let caps=Object.entries(r.capabilities).map(([k,v])=>`<span class="cap ${{v.sampled?'yes':v.evidence_supported?'ready':''}}">
${{esc(k)}} · ${{v.sampled?'已采样':v.evidence_supported?'可生成':'缺证据'}}</span>`).join('');
let tasks=r.reasoning_tasks?.length?`<details class="tasks"><summary>${{r.reasoning_tasks.length}} 个可执行推理任务 · ${{esc(r.reasoning_task_coverage)}}</summary>
${{r.reasoning_tasks.map(t=>`<div class="task"><small>${{esc(t.task_type)}} · ${{esc(t.program)}} · certificate=${{esc(t.certificate_result)}}</small>
<q>${{esc(t.question_zh)}}</q><div class="answer">GT：${{esc(t.surface_answer_zh)}}</div></div>`).join('')}}</details>`:'';
return `<article data-scene-id="${{esc(r.job_id)}}"><div class="head"><div><h2>${{esc(r.scene_model)}}</h2><span class="pill">${{esc(r.classification)}}</span></div>
<strong class="${{esc(r.status)}}">${{esc(r.status)}}</strong></div><div class="views">${{views}}</div><div class="body"><div class="metrics">
${{metric('实体/已观察',`${{r.evidence.observed_entity_count??'—'}} / ${{r.entity_count||'—'}}`)}}
${{metric('类别/区域',`${{r.evidence.observed_category_count??'—'}} / ${{r.evidence.observed_region_count??'—'}}`)}}
${{metric('跨视图候选',r.evidence.cross_view_relation_pair_count)}}
${{metric('最小可见实例',r.quality.minimum_visible_instance_count)}}
${{metric('最小有效深度',r.quality.minimum_valid_depth_fraction)}}
${{metric('训练证据门',r.evidence_eligibility)}}
${{metric('闭环结构相关',r.quality.loop_closure?.rgb_structure_correlation)}}
${{metric('去曝光 PSNR',r.quality.loop_closure?.bias_corrected_rgb_psnr_db)}}</div><div class="caps">${{caps||'<span class="empty">等待 reasoning audit</span>'}}</div>
${{tasks}}
${{r.preview_html?`<a href="../${{esc(r.preview_html)}}" target="_blank">查看全部 RGB / depth / mask ↗</a>`:''}}
<div class="decision">${{['accept','review','reject'].map(x=>`<button data-id="${{r.job_id}}" data-d="${{x}}" class="${{d.decision===x?'active':''}}">${{{{accept:'通过',review:'复核',reject:'拒绝'}}[x]}}</button>`).join('')}}</div>
<textarea data-note="${{r.job_id}}" placeholder="审阅备注">${{esc(d.note)}}</textarea></div></article>`}}).join('');
document.querySelectorAll('[data-d]').forEach(b=>b.onclick=()=>{{let id=b.dataset.id; decisions[id]={{...(decisions[id]||{{}}),decision:b.dataset.d,updated_at_utc:new Date().toISOString()}};
localStorage.setItem(key,JSON.stringify(decisions));render()}}); document.querySelectorAll('[data-note]').forEach(t=>t.onchange=()=>{{let id=t.dataset.note;
decisions[id]={{...(decisions[id]||{{decision:'unreviewed'}}),note:t.value,updated_at_utc:new Date().toISOString()}};
localStorage.setItem(key,JSON.stringify(decisions));render()}})}}
document.querySelector('#search').oninput=render;status.onchange=render;
document.querySelector('#next').onclick=()=>{{status.value='';document.querySelector('#search').value='';render();let target=data.records.find(r=>!['accept','review','reject'].includes(decisions[r.job_id]?.decision));if(target)requestAnimationFrame(()=>document.querySelector(`[data-scene-id="${{CSS.escape(target.job_id)}}"]`)?.scrollIntoView({{behavior:'smooth',block:'start'}}))}};
document.querySelector('#import').onchange=async e=>{{let file=e.target.files[0];if(!file)return;try{{let payload=JSON.parse(await file.text());
if(payload.schema_version!=='omnigibson_human_review.v1'||payload.sweep_id!==data.sweep_id)throw new Error('schema 或 sweep_id 不匹配');
decisions=payload.decisions||{{}};reviewer.value=payload.reviewer||'';localStorage.setItem(key,JSON.stringify(decisions));
localStorage.setItem(reviewerKey,reviewer.value.trim());render()}}catch(err){{alert('导入失败：'+err.message)}}}};
document.querySelector('#export').onclick=()=>{{let reviewerId=reviewer.value.trim();if(!reviewerId){{alert('请先填写审阅人');reviewer.focus();return}}let remaining=data.records.filter(r=>!['accept','review','reject'].includes(decisions[r.job_id]?.decision)).length;if(remaining&&!confirm(`仍有 ${{remaining}} 个场景未审。是否导出检查点 JSON？`))return;
localStorage.setItem(reviewerKey,reviewerId);let blob=new Blob([
JSON.stringify({{schema_version:'omnigibson_human_review.v1',sweep_id:data.sweep_id,reviewer:reviewerId,
generated_at_utc:new Date().toISOString(),decisions}},null,2)],{{type:'application/json'}}),a=document.createElement('a');
a.href=URL.createObjectURL(blob);a.download=data.sweep_id+'.human-review.json';a.click();URL.revokeObjectURL(a.href)}};render();
</script></html>"""

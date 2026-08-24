"""Build the pilot spot-check page: sampled question groups with gold labels.

Samples question groups from a coverage output, stratified by capability, and
writes ``<output-root>/review/index.html`` whose rows link to each family's
existing review viewer via relative paths.

    .venv/bin/python scripts/make_pilot_review.py \
        --output-root outputs/p23_pilot_v1/output --count 20
"""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path


def _load_groups(output_root: Path) -> list[dict]:
    rows = []
    for group_json in sorted((output_root / "groups").glob("*/group.json")):
        group = json.loads(group_json.read_text(encoding="utf-8"))
        group_dir = group_json.parent
        media = sorted(group_dir.glob("media/view-*.rgb.png"))
        rows.append(
            {
                "group_dir": group_dir,
                "group_id": group["question_group_id"],
                "scene": group_dir.name.split("__")[0],
                "cell_capability": group_dir.name.split("__")[1]
                if "__" in group_dir.name
                else "?",
                "questions": group["questions"],
                "media": media,
            }
        )
    return rows


def _sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    by_capability: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_capability[row["cell_capability"]].append(row)
    rng = random.Random(seed)
    for bucket in by_capability.values():
        rng.shuffle(bucket)
    picked: list[dict] = []
    buckets = sorted(by_capability)
    while len(picked) < count and any(by_capability[key] for key in buckets):
        for key in buckets:
            if by_capability[key] and len(picked) < count:
                picked.append(by_capability[key].pop())
    return picked


def _question_cells(review_dir: Path, row: dict) -> str:
    parts = []
    for question in row["questions"]:
        capability = question["capability"]
        label = question["label"]
        skip = question["skip_reason"]
        family_page = row["group_dir"] / capability / "index.html"
        link = html.escape(
            str(family_page.relative_to(review_dir.parent)), quote=True
        )
        if skip is None:
            badge = f'<span class="gold">{html.escape(str(label))}</span>'
        else:
            badge = f'<span class="skip" title="{html.escape(str(skip))}">跳过</span>'
        parts.append(
            f'<div class="q"><a href="../{link}">{html.escape(capability)}</a> {badge}</div>'
        )
    return "".join(parts)


def _thumbs(review_dir: Path, row: dict) -> str:
    frames = row["media"]
    if len(frames) > 6:
        step = (len(frames) - 1) / 5
        frames = [frames[round(i * step)] for i in range(6)]
    cells = []
    for frame in frames:
        rel = html.escape(str(frame.relative_to(review_dir.parent)), quote=True)
        cells.append(f'<img src="../{rel}" loading="lazy" alt="">')
    return "".join(cells)


def build_page(output_root: Path, count: int, seed: int) -> Path:
    rows = _load_groups(output_root)
    picked = _sample(rows, count, seed)
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    body = []
    for order, row in enumerate(picked, start=1):
        body.append(
            "<tr>"
            f'<td class="num">{order}</td>'
            f"<td>{html.escape(row['scene'])}<br>"
            f'<span class="dim">{html.escape(row["group_dir"].name)}</span></td>'
            f"<td>{_question_cells(review_dir, row)}</td>"
            f'<td class="film">{_thumbs(review_dir, row)}</td>'
            "</tr>"
        )
    page = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>P2/P3 试产抽检 · {len(picked)} 组</title>
<style>
 body {{ font-family: system-ui, "PingFang SC", sans-serif; margin: 0;
        background: #f5f6f8; color: #1c1e21; }}
 header {{ background: #fff; border-bottom: 1px solid #e0e2e6; padding: 14px 24px; }}
 header h1 {{ margin: 0; font-size: 17px; }}
 .wrap {{ max-width: 1500px; margin: 0 auto; padding: 18px 24px 60px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }}
 th, td {{ border-bottom: 1px solid #eceef1; padding: 8px 10px; text-align: left;
          vertical-align: top; }}
 th {{ background: #fafbfc; color: #5a5e66; }}
 .dim {{ color: #8a8e96; font-size: 11px; }}
 .q {{ margin: 2px 0; white-space: nowrap; }}
 .gold {{ background: #1d7a2c; color: #fff; padding: 1px 8px; border-radius: 9px;
         font-weight: 600; }}
 .skip {{ background: #ececf0; color: #666; padding: 1px 8px; border-radius: 9px; }}
 .film img {{ width: 96px; height: 96px; object-fit: cover; border-radius: 4px;
             margin-right: 4px; }}
 .num {{ color: #5a5e66; }}
</style></head><body>
<header><h1>P2/P3 试产抽检 · 每行点开能力名进入 family 审核页核对 gold</h1></header>
<div class="wrap"><table>
<tr><th>#</th><th>场景 / cell</th><th>问题与 gold</th><th>轨迹帧(等距 6 帧)</th></tr>
{''.join(body)}
</table></div></body></html>"""
    out = review_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    out = build_page(args.output_root.resolve(), args.count, args.seed)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

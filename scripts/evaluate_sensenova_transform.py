#!/usr/bin/env python3
"""Run SenseNova-SI zero-shot evaluation on a Transform Pilot scope."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-mode", choices=("answer_only", "grounded_cot"), required=True
    )
    parser.add_argument(
        "--evaluation-scope",
        choices=("test", "full_test", "balanced_all", "all"),
        default="test",
        help="Select frozen test, unbalanced test, balanced all-split, or full corpus.",
    )
    parser.add_argument("--gpu-ids", default="1,2")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    from episode3d.transform_pilot.evaluation import evaluate_sensenova

    summary = evaluate_sensenova(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output_path=args.output,
        prompt_mode=args.prompt_mode,
        evaluation_scope=args.evaluation_scope,
        start_index=args.start_index,
        stop_index=args.stop_index,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

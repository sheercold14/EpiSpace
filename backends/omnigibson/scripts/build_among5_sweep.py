#!/usr/bin/env python3
"""Materialize the exact procedural Among-5 counterfactual sweep plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from omnigibson_episode.among5_sweep import build_among5_sweep_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = build_among5_sweep_plan(
        family_config_path=args.config,
        output_root=args.output_root,
        plan_path=args.plan,
    )
    print(
        f"plan={args.plan.resolve()} scenes={plan['scene_count']} "
        f"variants={plan['variant_count']} jobs={plan['job_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

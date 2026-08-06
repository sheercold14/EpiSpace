# Scripts index and retention policy

Policy (agreed 2026-08-06):

1. `scripts/` holds THIN entry points only: parse args, call `src/`, print.
   Logic used twice must sink into `src/` with tests.
2. A script stays at top level only while something references it (Makefile,
   docs, CI, tests) or it is active tooling for the current mainline.
3. Generators of released/pilot datasets are PROVENANCE: they move to
   `scripts/pilot/` when their datasets are frozen, never deleted while the
   datasets exist (a dataset without its generator is unreproducible).
4. Scratch/experiment code goes to a git-ignored `scratch/`, not here.

## Active (top level)

| Script | Role | Referenced by |
|---|---|---|
| project_status.py / project_doctor.py | repo status & health from manifests | Makefile, STATUS.md |
| build_transform_dataset.py | Transform Pilot dataset build | Makefile |
| build_qa_generation_pilot.py | QA generation pilot build | docs/qa_generation_pipeline.md |
| build_compute_matched_schedules.py | compute-matched training schedules | docs/compute_matching.md |
| train_qwen3vl_lora.py / predict_qwen3vl.py | training & inference entry points | docs/qwen3vl_pilot.md, tests |
| merge_qwen_predictions.py / build_qwen_pilot_subset.py / profile_qwen_schedules.py / build_qwen_weight_inventory.py | Qwen pilot tooling | tests |
| adjudicate_semantic_visual_audit.py / build_semantic_visual_audit_packet.py | semantic-visual audit flow | tests |
| build_web_candidate_catalog.py / build_web_release_catalog.py | review-web catalogs | tests |
| build_scriptgen_review.py | scriptgen trajectory review site | docs/scriptgen_engine.md workflow |

## Pilot provenance (`scripts/pilot/`)

Frozen generators of existing pilot datasets and reports. Kept for
reproducibility of the artifacts under `episode3D/data` and
`manifests/experiments`; not maintained, not extended.

| Script | Artifact it generated |
|---|---|
| build_scene_dialogues.py, dub_scene_dialogues.py | scene_dialogues_v1 |
| build_trajectory_dialogues.py | trajectory_dialogues_v2* |
| build_t10_compositions.py | t10_compositions_v2 |
| build_training_samples.py | training_samples.v1.jsonl |
| build_rsint_llm_episode.py | rsint_dialogue_pilot_v1/v2 |
| build_generalization_episode.py | generalization episode data |
| build_trajectory_catalog.py | web trajectory catalog |
| build_transform_preflight.py, build_transform_web.py | Transform Pilot artifacts |
| evaluate_sensenova_transform.py, merge_sensenova_predictions.py | frozen SenseNova baselines (manifests/experiments) |
| summarize_qwen_pilot.py, summarize_groupstep_pilot.py | pilot summary reports |

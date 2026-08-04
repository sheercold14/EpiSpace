.PHONY: test lint check doctor acquire-t3-pilot derive-t3-pilot inventory object-inventory split-manifest plan-sweep plan-sweep-m2 retry-sweep-m2 derive-m2 sweep-smoke review-m2 review-portal release-m2 verify-release-m2 freeze-release-m2 plan-typed-trajectories plan-t3-r0 plan-t3-r1 run-t3-r0 run-t3-r1 plan-t4 plan-t4-v2 run-t4-v2 plan-t4-v3 run-t4-v3 plan-t7 plan-t8 plan-t10 run-t4 run-t7 run-t8 run-t10 plan-interventions-m2 plan-model-swaps-m2 plan-interventions-m2-v2 plan-model-swaps-m2-v2 plan-model-swaps-m2-v3 plan-model-swaps-m2-v4 plan-model-swaps-m2-v5 run-interventions-m2-v2 run-model-swaps-m2-v5 intervention-smoke model-swap-smoke review-interventions-m2 review-model-swaps-m2 review-interventions-m2-v2 review-model-swaps-m2-v5 release-interventions-m2 release-model-swaps-m2 release-interventions-m2-v2 release-model-swaps-m2-v5 freeze-interventions-m2-v2 freeze-model-swaps-m2-v5 verify-interventions-release-m2 verify-model-swaps-release-m2 verify-interventions-release-m2-v2 verify-model-swaps-release-m2-v5

PYTHON ?= ../habitat/.conda/core/bin/python
GPU_IDS ?= 0
WORKERS ?= 1

test:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m pytest -q

lint:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m ruff check src tests
	@for script in scripts/*.sh; do bash -n "$$script"; done

check: lint test

doctor:
	bash scripts/run_in_omnigibson.sh python -m omnigibson_episode.cli doctor

acquire-t3-pilot:
	bash scripts/run_in_omnigibson.sh --accept-eula \
		python -m omnigibson_episode.cli acquire \
		--recipe configs/omnigibson_t3_rotation_station_pilot.yaml \
		--output outputs/sweeps/trajectory-pilot-v1/bundles/Rs_int_t3_seed17 \
		--gpu-id 0 --headless

derive-t3-pilot:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli compile \
		--bundle outputs/sweeps/trajectory-pilot-v1/bundles/Rs_int_t3_seed17 \
		--output outputs/sweeps/trajectory-pilot-v1/bundles/Rs_int_t3_seed17/spatial_episode.json
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli inspect \
		--bundle outputs/sweeps/trajectory-pilot-v1/bundles/Rs_int_t3_seed17

inventory:
	PYTHONPATH=src $(PYTHON) -m omnigibson_episode.cli inventory \
		--assets-root .data/omnigibson/behavior-1k-assets \
		--output dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json

object-inventory:
	PYTHONPATH=src $(PYTHON) -m omnigibson_episode.cli object-inventory \
		--assets-root .data/omnigibson/behavior-1k-assets \
		--output outputs/catalogs/behavior-1k-v3.9.0-object-models.v1.json \
		--summary-output dataset/manifests/behavior-1k-v3.9.0-object-model-summary.v1.json

split-manifest: inventory
	PYTHONPATH=src $(PYTHON) -m omnigibson_episode.cli split-manifest \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--output dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json

plan-sweep: inventory
	PYTHONPATH=src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--base-recipe configs/omnigibson_static_m1.yaml \
		--output-root outputs/sweeps/static-m1-indoor-seed17-v1 \
		--output outputs/sweeps/static-m1-indoor-seed17-v1/sweep_plan.json

plan-sweep-m2: inventory
	PYTHONPATH=src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--base-recipe configs/omnigibson_static_m2_room_aware.yaml \
		--scene-overrides configs/omnigibson_static_m2_scene_overrides.yaml \
		--output-root outputs/sweeps/static-m2-room-aware-indoor-seed17-v1 \
		--output outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json

retry-sweep-m2: plan-sweep-m2
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--gpu-ids 0,1,2,3 --workers 4 --retry-failed \
		--acquire-timeout-minutes 40

derive-m2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli refresh-derived \
		--plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json

sweep-smoke: plan-sweep
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/static-m1-indoor-seed17-v1/sweep_plan.json \
		--gpu-ids 0 --workers 1 --limit 1

plan-typed-trajectories: plan-t4 plan-t7 plan-t8 plan-t10

plan-t3-r0:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--base-recipe configs/omnigibson_t3_rotation_station_pilot.yaml \
		--output-root outputs/sweeps/t3-rotation-station-r0-seed17-v1 \
		--output outputs/sweeps/t3-rotation-station-r0-seed17-v1/sweep_plan.json

plan-t3-r1:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--base-recipe configs/omnigibson_t3_rotation_station_rank1.yaml \
		--output-root outputs/sweeps/t3-rotation-station-r1-seed23-v1 \
		--output outputs/sweeps/t3-rotation-station-r1-seed23-v1/sweep_plan.json

plan-t4:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--base-recipe configs/omnigibson_t4_object_orbit.yaml \
		--output-root outputs/sweeps/t4-object-orbit-seed17-v1 \
		--output outputs/sweeps/t4-object-orbit-seed17-v1/sweep_plan.json

plan-t4-v2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--base-recipe configs/omnigibson_t4_object_orbit_adaptive_arc.yaml \
		--output-root outputs/sweeps/t4-object-orbit-adaptive-arc-seed17-v2 \
		--output outputs/sweeps/t4-object-orbit-adaptive-arc-seed17-v2/sweep_plan.json

plan-t4-v3:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--base-recipe configs/omnigibson_t4_object_orbit_adaptive_arc_v3.yaml \
		--output-root outputs/sweeps/t4-object-orbit-adaptive-arc-seed17-v3 \
		--output outputs/sweeps/t4-object-orbit-adaptive-arc-seed17-v3/sweep_plan.json

plan-t7:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--base-recipe configs/omnigibson_t7_elevation.yaml \
		--output-root outputs/sweeps/t7-elevation-seed17-v1 \
		--output outputs/sweeps/t7-elevation-seed17-v1/sweep_plan.json

plan-t8:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--base-recipe configs/omnigibson_t8_occlusion_reveal.yaml \
		--output-root outputs/sweeps/t8-occlusion-reveal-seed17-v1 \
		--output outputs/sweeps/t8-occlusion-reveal-seed17-v1/sweep_plan.json

plan-t10:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-sweep \
		--inventory dataset/manifests/behavior-1k-v3.9.0-scenes.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--base-recipe configs/omnigibson_t10_target_view.yaml \
		--output-root outputs/sweeps/t10-target-view-seed17-v1 \
		--output outputs/sweeps/t10-target-view-seed17-v1/sweep_plan.json

run-t4: plan-t4
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/t4-object-orbit-seed17-v1/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 40

run-t3-r0: plan-t3-r0
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/t3-rotation-station-r0-seed17-v1/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 40

run-t3-r1: plan-t3-r1
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/t3-rotation-station-r1-seed23-v1/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 40

run-t4-v2: plan-t4-v2
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/t4-object-orbit-adaptive-arc-seed17-v2/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 40

run-t4-v3: plan-t4-v3
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/t4-object-orbit-adaptive-arc-seed17-v3/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 40

run-t7: plan-t7
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/t7-elevation-seed17-v1/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 40

run-t8: plan-t8
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/t8-occlusion-reveal-seed17-v1/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 40

run-t10: plan-t10
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-sweep \
		--plan outputs/sweeps/t10-target-view-seed17-v1/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 40

review-m2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli review-sweep \
		--plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json

review-portal:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli review-portal \
		--static-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--intervention-plan outputs/sweeps/relation-flip-m2-v1/sweep_plan.json \
		--intervention-plan outputs/sweeps/model-swap-m2-v1/sweep_plan.json \
		--intervention-plan outputs/sweeps/relation-flip-m2-room-safe-v2/sweep_plan.json \
		--intervention-plan outputs/sweeps/model-swap-m2-conservative-v2/sweep_plan.json \
		--intervention-plan outputs/sweeps/model-swap-m2-clearance-v3/sweep_plan.json \
		--intervention-plan outputs/sweeps/model-swap-m2-strict-clearance-v4/sweep_plan.json \
		--intervention-plan outputs/sweeps/model-swap-m2-kinematic-v5/sweep_plan.json \
		--output outputs/review/index.html

release-m2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli build-release \
		--sweep-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--output dataset/releases/episode3d-og-static-m2-working.json \
		--release-id episode3d-og-static-m2-working

verify-release-m2: release-m2
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli verify-release \
		--manifest dataset/releases/episode3d-og-static-m2-working.json \
		--data-root outputs/sweeps/static-m2-room-aware-indoor-seed17-v1

freeze-release-m2:
	@test -n "$(HUMAN_REVIEW)" || (echo "HUMAN_REVIEW=<review.json> is required" >&2; exit 2)
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli build-release \
		--sweep-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--human-review $(HUMAN_REVIEW) \
		--output dataset/releases/episode3d-og-static-m2-v1.json \
		--release-id episode3d-og-static-m2-v1 \
		--freeze

plan-interventions-m2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-intervention-sweep \
		--static-sweep-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--object-inventory outputs/catalogs/behavior-1k-v3.9.0-object-models.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--output-root outputs/sweeps/relation-flip-m2-v1 \
		--output outputs/sweeps/relation-flip-m2-v1/sweep_plan.json \
		--maximum-candidates-per-scene 3

plan-model-swaps-m2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-intervention-sweep \
		--static-sweep-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--object-inventory outputs/catalogs/behavior-1k-v3.9.0-object-models.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--output-root outputs/sweeps/model-swap-m2-v1 \
		--output outputs/sweeps/model-swap-m2-v1/sweep_plan.json \
		--maximum-candidates-per-scene 3 \
		--intervention-kind model_swap

plan-interventions-m2-v2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-intervention-sweep \
		--static-sweep-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--object-inventory outputs/catalogs/behavior-1k-v3.9.0-object-models.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--output-root outputs/sweeps/relation-flip-m2-room-safe-v2 \
		--output outputs/sweeps/relation-flip-m2-room-safe-v2/sweep_plan.json \
		--sweep-id relation-flip-m2-room-safe-v2 \
		--maximum-candidates-per-scene 3

plan-model-swaps-m2-v2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-intervention-sweep \
		--static-sweep-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--object-inventory outputs/catalogs/behavior-1k-v3.9.0-object-models.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--output-root outputs/sweeps/model-swap-m2-conservative-v2 \
		--output outputs/sweeps/model-swap-m2-conservative-v2/sweep_plan.json \
		--sweep-id model-swap-m2-conservative-v2 \
		--maximum-candidates-per-scene 3 \
		--intervention-kind model_swap

plan-model-swaps-m2-v3:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-intervention-sweep \
		--static-sweep-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--object-inventory outputs/catalogs/behavior-1k-v3.9.0-object-models.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--output-root outputs/sweeps/model-swap-m2-clearance-v3 \
		--output outputs/sweeps/model-swap-m2-clearance-v3/sweep_plan.json \
		--sweep-id model-swap-m2-clearance-v3 \
		--maximum-candidates-per-scene 3 \
		--intervention-kind model_swap

plan-model-swaps-m2-v4:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-intervention-sweep \
		--static-sweep-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--object-inventory outputs/catalogs/behavior-1k-v3.9.0-object-models.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--output-root outputs/sweeps/model-swap-m2-strict-clearance-v4 \
		--output outputs/sweeps/model-swap-m2-strict-clearance-v4/sweep_plan.json \
		--sweep-id model-swap-m2-strict-clearance-v4 \
		--maximum-candidates-per-scene 3 \
		--intervention-kind model_swap \
		--model-swap-maximum-baseline-overlap-count 0 \
		--model-swap-minimum-nonoverlap-clearance-m 0.15

plan-model-swaps-m2-v5:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli plan-intervention-sweep \
		--static-sweep-plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
		--object-inventory outputs/catalogs/behavior-1k-v3.9.0-object-models.v1.json \
		--split-manifest dataset/manifests/behavior-1k-v3.9.0-scene-splits.v1.json \
		--output-root outputs/sweeps/model-swap-m2-kinematic-v5 \
		--output outputs/sweeps/model-swap-m2-kinematic-v5/sweep_plan.json \
		--sweep-id model-swap-m2-kinematic-v5 \
		--maximum-candidates-per-scene 3 \
		--intervention-kind model_swap \
		--model-swap-maximum-baseline-overlap-count 0 \
		--model-swap-minimum-nonoverlap-clearance-m 0.15 \
		--model-swap-execution-mode kinematic_counterfactual \
		--intervention-settle-steps 0

run-interventions-m2-v2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-intervention-sweep \
		--plan outputs/sweeps/relation-flip-m2-room-safe-v2/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 60

run-model-swaps-m2-v5:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-intervention-sweep \
		--plan outputs/sweeps/model-swap-m2-kinematic-v5/sweep_plan.json \
		--gpu-ids $(GPU_IDS) --workers $(WORKERS) --acquire-timeout-minutes 60

intervention-smoke: plan-interventions-m2
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-intervention-sweep \
		--plan outputs/sweeps/relation-flip-m2-v1/sweep_plan.json \
		--gpu-ids 0 --workers 1 --limit 1

model-swap-smoke: plan-model-swaps-m2
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli run-intervention-sweep \
		--plan outputs/sweeps/model-swap-m2-v1/sweep_plan.json \
		--gpu-ids 0 --workers 1 --limit 1

review-interventions-m2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli review-interventions \
		--plan outputs/sweeps/relation-flip-m2-v1/sweep_plan.json

review-model-swaps-m2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli review-interventions \
		--plan outputs/sweeps/model-swap-m2-v1/sweep_plan.json

review-interventions-m2-v2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli review-interventions \
		--plan outputs/sweeps/relation-flip-m2-room-safe-v2/sweep_plan.json

review-model-swaps-m2-v5:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli review-interventions \
		--plan outputs/sweeps/model-swap-m2-kinematic-v5/sweep_plan.json

release-interventions-m2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli build-pair-release \
		--plan outputs/sweeps/relation-flip-m2-v1/sweep_plan.json \
		--output dataset/releases/episode3d-og-relation-flip-m2-working.json \
		--release-id episode3d-og-relation-flip-m2-working

release-model-swaps-m2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli build-pair-release \
		--plan outputs/sweeps/model-swap-m2-v1/sweep_plan.json \
		--output dataset/releases/episode3d-og-model-swap-m2-working.json \
		--release-id episode3d-og-model-swap-m2-working

release-interventions-m2-v2:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli build-pair-release \
		--plan outputs/sweeps/relation-flip-m2-room-safe-v2/sweep_plan.json \
		--output dataset/releases/episode3d-og-relation-flip-m2-v2-working.json \
		--release-id episode3d-og-relation-flip-m2-v2-working

release-model-swaps-m2-v5:
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli build-pair-release \
		--plan outputs/sweeps/model-swap-m2-kinematic-v5/sweep_plan.json \
		--output dataset/releases/episode3d-og-model-swap-m2-v5-working.json \
		--release-id episode3d-og-model-swap-m2-v5-working

freeze-interventions-m2-v2:
	@test -n "$(HUMAN_REVIEW)" || (echo "HUMAN_REVIEW=<pair-review.json> is required" >&2; exit 2)
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli build-pair-release \
		--plan outputs/sweeps/relation-flip-m2-room-safe-v2/sweep_plan.json \
		--human-review $(HUMAN_REVIEW) \
		--output dataset/releases/episode3d-og-relation-flip-m2-v2.json \
		--release-id episode3d-og-relation-flip-m2-v2 --freeze

freeze-model-swaps-m2-v5:
	@test -n "$(HUMAN_REVIEW)" || (echo "HUMAN_REVIEW=<pair-review.json> is required" >&2; exit 2)
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli build-pair-release \
		--plan outputs/sweeps/model-swap-m2-kinematic-v5/sweep_plan.json \
		--human-review $(HUMAN_REVIEW) \
		--output dataset/releases/episode3d-og-model-swap-m2-v5.json \
		--release-id episode3d-og-model-swap-m2-v5 --freeze

verify-interventions-release-m2: release-interventions-m2
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli verify-pair-release \
		--manifest dataset/releases/episode3d-og-relation-flip-m2-working.json \
		--static-data-root outputs/sweeps/static-m2-room-aware-indoor-seed17-v1 \
		--intervention-data-root outputs/sweeps/relation-flip-m2-v1

verify-model-swaps-release-m2: release-model-swaps-m2
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli verify-pair-release \
		--manifest dataset/releases/episode3d-og-model-swap-m2-working.json \
		--static-data-root outputs/sweeps/static-m2-room-aware-indoor-seed17-v1 \
		--intervention-data-root outputs/sweeps/model-swap-m2-v1

verify-interventions-release-m2-v2: release-interventions-m2-v2
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli verify-pair-release \
		--manifest dataset/releases/episode3d-og-relation-flip-m2-v2-working.json \
		--static-data-root outputs/sweeps/static-m2-room-aware-indoor-seed17-v1 \
		--intervention-data-root outputs/sweeps/relation-flip-m2-room-safe-v2

verify-model-swaps-release-m2-v5: release-model-swaps-m2-v5
	PYTHONPATH=src:../habitat/src $(PYTHON) -m omnigibson_episode.cli verify-pair-release \
		--manifest dataset/releases/episode3d-og-model-swap-m2-v5-working.json \
		--static-data-root outputs/sweeps/static-m2-room-aware-indoor-seed17-v1 \
		--intervention-data-root outputs/sweeps/model-swap-m2-kinematic-v5

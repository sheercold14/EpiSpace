PYTHON ?= python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff

.PHONY: install doctor status quickstart check test test-core test-backend test-artifact web reproduce-transform-pilot

install:
	$(PYTHON) -m pip install -e ".[dev]"

doctor:
	$(PYTHON) scripts/project_doctor.py

status:
	$(PYTHON) scripts/project_status.py

quickstart: doctor
	$(PYTEST) -q \
		tests/test_transform_pilot_geometry.py \
		tests/test_transform_evaluation.py \
		backends/omnigibson/tests/test_geometry.py

check:
	$(RUFF) check src backends/omnigibson/src scripts tests backends/omnigibson/tests

# Portable tests that do not require private trajectory bundles or simulator assets.
test-core:
	$(PYTEST) -q \
		tests/test_transform_pilot_geometry.py \
		tests/test_transform_evaluation.py

test-backend:
	cd backends/omnigibson && $(abspath $(PYTHON)) -m pytest -q tests

test: test-core test-backend

# Full replay suite. Connect the external data workspace before invoking it.
test-artifact:
	$(PYTEST) -q tests

web:
	$(PYTHON) -m http.server 8770 --bind 127.0.0.1 --directory web

# Requires workspace.env paths and the copied source trajectory bundles.
reproduce-transform-pilot:
	@test -f workspace.env || (echo "copy workspace.example.env to workspace.env first"; exit 2)
	set -a; . ./workspace.env; set +a; \
	$(PYTHON) scripts/build_transform_dataset.py --config configs/transform_dataset_v1.json

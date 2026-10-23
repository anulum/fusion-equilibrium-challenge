# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — declared verification and reproduction commands
TOOLBIN ?= .venv/bin
PYTHON ?= $(TOOLBIN)/python
CONSTRUCTIVE_TOOLBIN ?= .venv-constructive/bin
CONSTRUCTIVE_PYTHON ?= $(CONSTRUCTIVE_TOOLBIN)/python
CONSTRUCTIVE_REFERENCE_PYTHON ?= .venv-constructive-reference/bin/python
CONSTRUCTIVE_WRONG_VERSION_PYTHON ?= .venv-constructive-wrong-version/bin/python
CONSTRUCTIVE_COVERAGE_FILE ?= .coverage-constructive
CONSTRUCTIVE_SOURCES := $(wildcard tools/constructive_*.py)
CONSTRUCTIVE_TESTS := $(wildcard tests/test_constructive_*.py)
TESTS := $(filter-out $(CONSTRUCTIVE_TESTS),$(wildcard tests/test_*.py))
SOURCES := experiments/d71_finisher_physics/per_shot_table.py tools/check_doc_links.py tools/ci_check_shot.py tools/d71b_table.py tools/d72_table.py tools/surface_gs.py tools/surface_support_summary.py tools/surface_support_table.py tools/scored_map_summary.py tools/scored_map_table.py tools/audit_map_input.py tools/field_diagnostics.py tools/equilibrium_audit.py tools/competition_map.py tools/official_ridge.py tools/audit_calibration.py tools/make_preview.py tools/check_publication_boundary.py
FROZEN := source training_adapter infer.py train.py tree_replica.py deployment.json training_recipe.json WEIGHTS.json WEIGHTS.sha256
FINAL := 25ba6fd4257cc37451f851983d8eaeb7e7381e02

.PHONY: check quality constructive-quality links frozen test constructive-test boundary
check: quality constructive-quality links frozen boundary

quality:
	$(TOOLBIN)/ruff check $(SOURCES) $(TESTS)
	$(TOOLBIN)/ruff format --check $(SOURCES) $(TESTS)
	$(TOOLBIN)/mypy
	$(TOOLBIN)/bandit -q $(SOURCES)
	$(TOOLBIN)/reuse lint

constructive-quality:
	$(TOOLBIN)/ruff check $(CONSTRUCTIVE_SOURCES) $(CONSTRUCTIVE_TESTS)
	$(TOOLBIN)/ruff format --check $(CONSTRUCTIVE_SOURCES) $(CONSTRUCTIVE_TESTS)
	$(CONSTRUCTIVE_TOOLBIN)/mypy --strict $(CONSTRUCTIVE_SOURCES) $(CONSTRUCTIVE_TESTS)
	$(TOOLBIN)/bandit -q $(CONSTRUCTIVE_SOURCES)

links:
	$(PYTHON) tools/check_doc_links.py

frozen:
	test "$$(git rev-parse 'final-submission-2026-10-19^{commit}')" = "$(FINAL)"
	git diff --exit-code $(FINAL) -- $(FROZEN)

test:
	$(PYTHON) -m coverage run --parallel-mode -m pytest -q $(TESTS)
	$(PYTHON) -m coverage combine
	$(PYTHON) -m coverage report

constructive-test:
	test -x "$(CONSTRUCTIVE_REFERENCE_PYTHON)"
	test -x "$(CONSTRUCTIVE_WRONG_VERSION_PYTHON)"
	CONSTRUCTIVE_REFERENCE_PYTHON="$(CONSTRUCTIVE_REFERENCE_PYTHON)" CONSTRUCTIVE_WRONG_VERSION_PYTHON="$(CONSTRUCTIVE_WRONG_VERSION_PYTHON)" COVERAGE_FILE="$(CONSTRUCTIVE_COVERAGE_FILE)" $(CONSTRUCTIVE_PYTHON) -m coverage run --branch --parallel-mode --source=tools -m pytest -q -rs $(CONSTRUCTIVE_TESTS)
	COVERAGE_FILE="$(CONSTRUCTIVE_COVERAGE_FILE)" $(CONSTRUCTIVE_PYTHON) -m coverage combine --keep
	COVERAGE_FILE="$(CONSTRUCTIVE_COVERAGE_FILE)" $(CONSTRUCTIVE_PYTHON) -m coverage report --include='tools/constructive_*.py'

boundary:
	$(PYTHON) tools/check_publication_boundary.py

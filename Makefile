# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Fusion Equilibrium Challenge — declared verification and reproduction commands
TOOLBIN ?= .venv/bin
PYTHON ?= $(TOOLBIN)/python
SOURCES := experiments/d71_finisher_physics/per_shot_table.py tools/check_doc_links.py tools/ci_check_shot.py
FROZEN := source training_adapter infer.py train.py tree_replica.py deployment.json training_recipe.json WEIGHTS.json WEIGHTS.sha256
FINAL := 19d2d341f3498617d6b28b5b90718a5d0858c859

.PHONY: check quality links frozen test
check: quality links frozen

quality:
	$(TOOLBIN)/ruff check $(SOURCES) tests
	$(TOOLBIN)/ruff format --check $(SOURCES) tests
	$(TOOLBIN)/mypy
	$(TOOLBIN)/bandit -q $(SOURCES)
	$(TOOLBIN)/reuse lint

links:
	$(PYTHON) tools/check_doc_links.py

frozen:
	test "$$(git rev-parse 'final-submission-2026-10-19^{commit}')" = "$(FINAL)"
	git diff --exit-code $(FINAL) -- $(FROZEN)

test:
	$(PYTHON) -m coverage run --parallel-mode -m pytest -q tests
	$(PYTHON) -m coverage combine
	$(PYTHON) -m coverage report

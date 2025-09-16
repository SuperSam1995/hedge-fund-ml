.PHONY: setup lint test data train_ae train_gan replicate eval report reproduce

UV ?= uv
PYTHON := $(UV) run --group dev python
PYTHON_DL := $(UV) run --group dev --extra deep-learning python

setup:
	$(UV) sync --group dev

lint:
	$(UV) run --group dev ruff check src tests
	$(UV) run --group dev black --check src tests
	$(UV) run --group dev mypy src tests

test:
	$(UV) run --group dev pytest -q

data:
	$(PYTHON) scripts/data_prepare.py

train_ae:
	$(PYTHON_DL) -m hedge_fund_ml.cli train-ae

train_gan:
	$(PYTHON_DL) -m hedge_fund_ml.cli train-gan

replicate:
	$(PYTHON) -m hedge_fund_ml.cli replicate

eval:
	$(PYTHON) -m hedge_fund_ml.cli eval

report:
	$(PYTHON) -m hedge_fund_ml.cli report

reproduce:
	$(PYTHON_DL) -m hedge_fund_ml.cli reproduce

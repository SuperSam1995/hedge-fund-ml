.PHONY: setup lint test data features train_ae train_gan replicate eval report reproduce

UV ?= uv
PYTHON := $(UV) run --group dev python
PYTHON_DL := $(UV) run --group dev --extra deep-learning python
REPORT_DATE := $(shell date +%Y-%m-%d)

setup:
	$(UV) sync --group dev

lint:
	$(UV) run --group dev ruff check src tests
	$(UV) run --group dev black --check src tests
	$(UV) run --group dev mypy src tests

test:
	$(UV) run --group dev --extra deep-learning pytest -q

data:
	$(PYTHON) scripts/data_prepare.py

features:
	$(PYTHON) -m scripts.build_features --config configs/features.yaml

train_ae:
	$(PYTHON_DL) -m hedge_fund_ml.cli train-ae

train_gan:
	$(PYTHON_DL) -m hedge_fund_ml.cli train-gan

replicate:
	$(PYTHON) -m scripts.replicate --config configs/replicate.yaml

eval:
	$(PYTHON) -m scripts.eval --config configs/eval.yaml

report:
	$(UV) run --group dev papermill notebooks/final_report.ipynb reports/_tmp_$(REPORT_DATE).ipynb \
	  -p metrics_path reports/metrics_latest.json \
	  -p figures_dir reports/figures
	$(UV) run --group dev jupyter nbconvert --to html --no-input reports/_tmp_$(REPORT_DATE).ipynb \
	  --output-dir reports \
	  --output final_report_$(REPORT_DATE).html

reproduce:
	$(PYTHON_DL) -m hedge_fund_ml.cli reproduce

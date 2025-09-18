.PHONY: setup lint test data features train_ae train_gan replicate eval report reproduce

UV ?= uv
PYTHON := $(UV) run --group dev python
PYTHON_DL := $(UV) run --group dev --extra deep-learning python
REPORT_DATE := $(shell date +%Y-%m-%d)
REPORT_NOTEBOOK := notebooks/final_report.py  # text notebook (py:percent)

setup:
	$(UV) sync --group dev

lint:
	uv run --group dev ruff check src tests
	uv run --group dev black --check src tests
	uv run --group dev mypy src tests

test:
	uv run --group dev --extra deep-learning pytest -q

data:
	uv run --group dev python -m scripts.data_prepare --config configs/data.yaml

features:
	uv run --group dev python -m scripts.build_features --config configs/features.yaml

train_ae:
	uv run --group dev --extra deep-learning python -m scripts.train_ae

train_gan:
	uv run --group dev --extra deep-learning python -m scripts.train_gan

replicate:
	uv run --group dev python -m scripts.replicate --config configs/replicate.yaml

eval:
	uv run --group dev python -m scripts.eval --config configs/eval.yaml

report:
	$(UV) run --group dev jupytext --to ipynb $(REPORT_NOTEBOOK) -o reports/_tmp_$(REPORT_DATE).ipynb
	$(UV) run --group dev papermill reports/_tmp_$(REPORT_DATE).ipynb reports/_exec_$(REPORT_DATE).ipynb \
	  -p metrics_path reports/metrics_latest.json \
	  -p figures_dir reports/figures
	$(UV) run --group dev jupyter nbconvert --to html --no-input reports/_exec_$(REPORT_DATE).ipynb \
	  --output reports/final_report_$(REPORT_DATE).html

reproduce:
	$(PYTHON_DL) -m hedge_fund_ml.cli reproduce

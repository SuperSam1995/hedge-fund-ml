.PHONY: setup lint test data features train_ae train_gan replicate eval report reproduce bundle

UV ?= uv
PYTHON := $(UV) run --group dev python
PYTHON_DL := $(UV) run --group dev --extra deep-learning python
REPORT_DATE := $(shell date +%Y-%m-%d)
REPORT_NOTEBOOK := notebooks/final_report.py  # text notebook (py:percent)
bundle_dir := $(shell date -u +results/logs/bundle/%Y-%m-%dT%H%MZ)

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
	  -p metrics_path results/metrics_latest.json \
	  -p figures_dir results/figures
	$(UV) run --group dev jupyter nbconvert --to html --no-input reports/_exec_$(REPORT_DATE).ipynb \
	  --output reports/final_report_$(REPORT_DATE).html

reproduce:
	$(PYTHON_DL) -m hedge_fund_ml.cli reproduce

bundle:
	uv run --group dev python -m scripts.bundle_report --out $(bundle_dir)
	@echo "Bundle ready at $(bundle_dir)"

train_itrafo:
	uv run --group dev --group deep-learning-torch python -m scripts.train_transformer --config configs/transformer.yaml

replicate_itrafo:
	uv run --group dev python -m scripts.replicate_itrafo --config configs/replicate_itrafo.yaml

eval_itrafo:
	uv run --group dev python -m scripts.eval --config configs/eval_itrafo.yaml

report_itrafo:
	$(MAKE) eval_itrafo && $(MAKE) report

# ---------- CPU-only reproducible pipeline (no TF required) ----------
METRICS_JSON := $(shell ls -1t reports/metrics/*.json 2>/dev/null | head -n1)

.PHONY: reproduce_cpu report_cpu open-report

reproduce_cpu:
	uv run --group dev python -m scripts.build_features --config configs/features.yaml
	uv run --group dev python -m scripts.replicate     --config configs/replicate.yaml
	uv run --group dev python -m scripts.eval          --config configs/eval.yaml
	@mkdir -p reports/metrics
	cp results/metrics/*.json reports/metrics/
	$(MAKE) report_cpu


report_cpu:
	@mkdir -p reports reports/figures
	uv run --group dev jupytext --to ipynb notebooks/final_report.py -o reports/_tmp.ipynb
	uv run --group dev papermill reports/_tmp.ipynb reports/_exec.ipynb \
	  -p metrics_path $(METRICS_JSON) \
	  -p figures_dir reports/figures
	uv run --group dev jupyter nbconvert --to html --no-input \
	  --output-dir reports --output final_report.html reports/_exec.ipynb
	@echo "Report ready: reports/final_report.html"

open-report:
	$(PYTHON) -c "import pathlib, webbrowser; webbrowser.open(pathlib.Path('reports/final_report.html').resolve().as_uri())"

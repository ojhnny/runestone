# Every target runs inside the uv-managed environment, so there is no
# "activate the venv first" step. Install uv once: https://docs.astral.sh/uv/

.PHONY: setup test lint pipeline report app dagster mlflow notebooks pdf clean-data

setup:            ## install everything (core + dashboard + MLflow + Dagster + dev tools)
	uv sync --all-extras

test:             ## unit + synthetic end-to-end tests (no real data needed)
	uv run pytest -q

lint:
	uv run ruff check .

pipeline:         ## raw exports -> every table, model and figure (~15 min on a laptop)
	uv run so-ell all

report:           ## re-draw figures and tables only (seconds)
	uv run so-ell report

app:              ## interactive Streamlit dashboard
	uv run streamlit run app/streamlit_app.py

dagster:          ## asset graph UI at http://localhost:3000
	uv run dagster dev -m so_ell.orchestration

mlflow:           ## experiment tracking UI at http://localhost:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

notebooks:        ## run the whole pipeline through the build notebooks (needs data/raw)
	for nb in notebooks/0*.ipynb; do uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $$nb || exit 1; done

pdf:              ## fill numbers from outputs/ and rebuild docs/design_report.pdf
	uv run python docs/fill_numbers.py docs/design_report.template.md docs/design_report.md
	uv run python docs/build_report.py

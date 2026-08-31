# cycle-tracking

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Quickstart

```bash
# Install dependencies
pipenv install

# Activate environment
pipenv shell

# Run data pipeline (raw → processed)
make data

```

## Project Organization

```
├── Makefile               <- Pipeline commands: make data, make train
├── Pipfile                <- Dependencies (use pipenv, not pip/uv/conda)
├── pyproject.toml         <- Package metadata and tool config (ruff)
├── data/
│   ├── raw/               <- Immutable source data
│   ├── interim/           <- Intermediate transformations
│   ├── processed/         <- Final datasets for modeling
│   └── external/          <- Third-party sources
├── notebooks/             <- Exploratory notebooks (not production code)
├── reports/figures/       <- Generated figures (qc, diagnostics/, results/)
├── models/                <- pkl/ (serialized models) + results/ (metrics, runs.jsonl)
└── src/
    ├── config.py          <- Paths, env vars, logger
    ├── dataset.py         <- Raw → interday.csv (load, merge, clean)
    ├── qc.py              <- interday.csv → interday_qc.csv (wear time + day filters)
    ├── features.py        <- interday_qc.csv → interday_fitbit.csv + interday_selfreports.csv + interday_hormones.csv
    ├── plots.py           <- Eval metrics (all datasets) + confusion matrix/SHAP for one dataset (--dataset)
    └── modeling/
        ├── classifiers.py <- Classifier pipelines + Optuna hyperparameter search
        ├── train.py       <- Grouped train/val/test training (logreg, XGBoost)
        ├── predict.py     <- Inference
        ├── diagnostics.py <- Model diagnostics
        └── reporting.py   <- Logging/reporting helpers
```

## Development

```bash
# Lint + format
ruff check --fix . && ruff format .

# Run individual pipeline steps
pipenv run python src/dataset.py
pipenv run python src/qc.py
pipenv run python src/features.py
pipenv run python -m src.modeling.train all
pipenv run python -m src.plots --dataset fitbit_hormones_selfreports
pipenv run python -m src.modeling.diagnostics gap
```

## Dataset

Lin, B., Li, J. Y., Kalani, K., Truong, K., & Mariakakis, A. (2025). mcPHASES: A Dataset of Physiological, Hormonal, and Self-reported Events and Symptoms for Menstrual Health Tracking with Wearables (version 1.0.0). _PhysioNet_. https://doi.org/10.13026/zx6a-2c81

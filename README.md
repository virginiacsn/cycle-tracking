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
├── reports/figures/       <- Generated figures
├── models/                <- Serialized models
└── src/
    ├── config.py          <- Paths, env vars, logger
    ├── dataset.py         <- Raw → processed pipeline (Typer CLI)
    ├── features.py        <- Feature engineering
    ├── plots.py           <- Visualizations
    └── modeling/
        ├── train.py       <- Model training
        └── predict.py     <- Inference
```

## Development

```bash
# Lint + format
ruff check --fix . && ruff format .

# Tests
pipenv run pytest tests/ --cov

# Data pipeline step
pipenv run python src/dataset.py
```

See `.context/plan.md` for current task status and `.rules/` for coding standards.

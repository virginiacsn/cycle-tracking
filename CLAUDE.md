# cycle-tracking Instructions

## Project Context
**Purpose:** Cycle tracking data analysis — processing, feature engineering, modeling, and visualization of cycle data
**Tech Stack:** Python 3.12, pandas, scikit-learn, matplotlib, numpy, jupyter, loguru, typer, ruff, tqdm
**Architecture:** Cookiecutter Data Science layout; `src/` flat package, data pipeline via Makefile targets

## Architecture Map
```
src/
├── config.py      # Paths, env vars, logger setup
├── dataset.py     # Raw → processed data pipeline (Typer CLI)
├── features.py    # Feature engineering
├── plots.py       # Visualizations
└── modeling/
    ├── train.py   # Model training
    └── predict.py # Inference
data/
├── raw/           # Immutable source data
├── interim/       # Intermediate transformations
├── processed/     # Final datasets for modeling
└── external/      # Third-party sources
notebooks/         # Exploratory Jupyter notebooks
```

## Environment Setup
```bash
# Install dependencies
pipenv install

# Activate environment
pipenv shell

# Run linting
ruff check --fix . && ruff format .

# Run tests
pytest tests/

# Build data pipeline
make data
```

## Development Workflow
1. **Check context:** Review .context/plan.md for current tasks
2. **Understand deeply:** Check .context/ideas.md for design decisions
3. **Research if needed:** Update .context/research.md with findings
4. **Branch:** `gh issue develop <issue-number>`
5. **Code:** Follow patterns (see .rules/ for standards)
6. **Test:** Real data only (see .rules/testing.md)
7. **Document failures:** Log in .context/scratch_history.md immediately
8. **Commit:** Atomic, <50 chars, no emojis
9. **PR:** Reference context and issue
10. **Code review:** Run `/review-pr` after creating PR (see .rules/code_review.md)

## [CRITICAL] Core Principles - Never Compromise

### [FUNDAMENTAL] NO MOCKS - Test Reality Only
- Use real data samples or skip tests entirely
- Ask user for sample data if needed
**Details:** .rules/testing.md

### Commits & Git
- Atomic commits, focused changes
- Messages <50 chars, no emojis, no AI attribution
- Feature branches for multi-step work
**Details:** .rules/git.md

### No Technical Debt Carried Forward
- Address ALL PR review findings
- Only skip genuine false positives or intended design choices
- Replace, don't deprecate
**Details:** .rules/code_review.md

### Documentation
- Examples > explanations
- README gets someone running in <5 minutes
**Details:** .rules/documentation.md

## [NEVER DO THIS]
- Never use mocks, stubs, or fake data in tests
- Never use `uv`, `conda`, or `virtualenv`; use **pipenv** for this project
- Never commit secrets, .env files, or credentials
- Never leave empty catch blocks or silent failures
- Never add backward-compatibility shims; replace directly
- Never add TODO without a linked issue
- Never use `os.path`; use `pathlib.Path` (already set up in config.py)

## Think Like a Senior Developer
- Keep the big picture in mind
- Document learnings in .context/scratch_history.md
- Extract patterns (3+ uses) into rules
**See:** .rules/self_improve.md for learning process

## [REFERENCE] Rules Directory

### Core Standards
- `.rules/testing.md` - Complete NO MOCK policy
- `.rules/self_improve.md` - Learning from projects
- `.rules/documentation.md` - MkDocs setup
- `.rules/code_review.md` - PR review toolkit and checklist

### Language & Tools
- `.rules/python.md` - Python standards (ruff, type hints, patterns)
- `.rules/ci_cd.md` - GitHub Actions setup
- `.rules/git.md` - Commit and branch conventions

## Context Files
- `.context/plan.md` - Current tasks and phases
- `.context/research.md` - Technical explorations
- `.context/ideas.md` - Design concepts
- `.context/scratch_history.md` - Failed attempts

## Quick Commands
```bash
# Lint + format
ruff check --fix . && ruff format .

# Run tests
pipenv run pytest tests/ --cov

# Run data pipeline step
pipenv run python src/dataset.py

# Jupyter
pipenv run jupyter lab
```

## Project-Specific Guidelines
- All paths should use constants from `src/config.py` (RAW_DATA_DIR, PROCESSED_DATA_DIR, etc.)
- Use `loguru` for all logging (already configured in config.py)
- Use `typer` for CLI entry points
- Notebooks are for exploration only; production code lives in `src/`

---
Remember: You're building maintainable data pipelines, not just writing scripts.
Check .rules/ for detailed guidance on any topic.

# Python Development Standards

## Version & Environment
- **Python 3.12** (pinned in pyproject.toml)
- **Package Manager:** pipenv (not uv, conda, or virtualenv)
- **Virtual Environment:** Managed by pipenv (`pipenv install`, `pipenv shell`)
- **Project Config:** `pyproject.toml` + `Pipfile`

## Quick Reference
```bash
# Install all dependencies
pipenv install

# Add a dependency
pipenv install pandas

# Add a dev dependency
pipenv install --dev pytest

# Activate shell
pipenv shell

# Run without activating
pipenv run pytest
pipenv run python src/dataset.py
```

## Code Style
- **Formatter:** `ruff format` (Black-compatible)
- **Linter:** `ruff check --fix --unsafe-fixes`
- **Type Checker:** `ty` (not mypy)
- **Line Length:** 88 characters (Black standard)
- **Imports:** Sorted by ruff (isort-compatible)

## Type Hints
- **Required for:** All public functions and methods
- **Example:**
```python
def process_data(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Process raw data into DataFrame."""
    ...
```

## Project Structure
```
project/
├── src/project/       # Source code
│   ├── __init__.py
│   └── module.py
├── tests/            # Real tests only
├── pyproject.toml    # Project config (UV + ruff + ty)
└── .gitignore
```

## Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
if [ -n "$files" ]; then
    uv run ruff check --fix --unsafe-fixes $files
    uv run ruff format $files
    git add $files
fi
```

## Common Patterns
- **Context Managers:** For resource management
- **Dataclasses:** For data structures
- **Pathlib:** For file operations (not os.path)
- **F-strings:** For string formatting

## Error Handling
```python
# Be specific with exceptions
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise  # Re-raise or handle appropriately

# Never do this:
# except Exception:
#     pass  # Silent failure
```

## Never Do This
- Never use `uv`; use `pipenv` for this project
- Never use `conda`, `virtualenv`, or `venv`; pipenv handles environments
- Never use bare `except:` or `except Exception: pass`
- Never use `os.path`; use `pathlib.Path`
- Never commit `.env` files or hardcoded secrets

## Documentation
- **Docstrings:** Google or NumPy style
- **Module docs:** At file top
- **Type hints:** Self-documenting code

---
*Pipenv for environments. Ruff for style. Real tests only.*

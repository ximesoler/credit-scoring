# credit-scoring

This repository contains code and notebooks for building and evaluating credit scoring models.

## Requirements

- Python 3.11 (use an exact 3.11.x interpreter). Verify with:
  ```bash
  python --version
  # -> Python 3.11.x
  ```

- Recommended: use pyenv or asdf to install/manage Python versions on macOS. Example using pyenv:
  ```bash
  brew install pyenv
  pyenv install 3.11.6
  pyenv local 3.11.6
  ```

## Setup

1. Install Poetry (see https://python-poetry.org/docs/#installation). On macOS you can run:
	```bash
	curl -sSL https://install.python-poetry.org | python3 -
	```

2. Install project dependencies:
	```bash
	poetry install
	```

3. (Optional) Activate the virtual environment:
	```bash
	poetry shell
	```

4. Run project commands inside the environment:
	```bash
	poetry run python -m <module>
	```

## Repository structure

Top-level layout:

- `pyproject.toml`, `poetry.lock` - project metadata and pinned dependencies managed by Poetry.
- `config/` - configuration files (e.g., `config.yaml`).
- `models/` - trained model artifacts (check `.gitignore` for rules).
- `nb/` - Jupyter notebooks used for data labeling, EDA, model development and experimentation. Notebooks in this repo:
  - `1.Data_labeling.ipynb`
  - `2.EDA_selected_sample.ipynb`
  - `3.Model_Development.ipynb`
  - `functions.py` (helper functions used by notebooks)
- `reports/` - generated reports and figures.
- `src/` - source code for the project. Example package:
  - `score_pd/` - contains Python package code (e.g., `__init__.py`).

## Quick start

1. Ensure Python 3.11 is active.
2. Install Poetry and dependencies (`poetry install`).
3. Start a shell and run notebooks or scripts:
	```bash
	poetry shell
	jupyter lab  # or jupyter notebook
	```

## Notes

- If you need a specific Python 3.11 patch version, pin it with pyenv/asdf and use `pyenv local` or similar.
- This README contains the minimal setup steps; consult `pyproject.toml` for exact dependency versions.

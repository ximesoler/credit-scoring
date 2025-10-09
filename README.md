# Credit Risk Scoring Model

## Requirements

- **Python**: 3.11.x (exact version required)
- **Package Manager**: Poetry

### Installing Python 3.11

Verify your Python version:
```bash
python --version
# Should output: Python 3.11.x
```

If you need to install Python 3.11, use pyenv:

```bash
# Install pyenv
brew install pyenv
# or follow: https://github.com/pyenv/pyenv#installation

# Install Python 3.11
pyenv install 3.11.6
pyenv local 3.11.6

# Verify
python --version
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/ximesoler/credit-scoring.git
cd credit-scoring
```

### 2. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -

# Verify installation
poetry --version
```

### 3. Install Dependencies

```bash
# Install all project dependencies
poetry install

# Activate virtual environment (optional)
poetry shell
```

### 4. Run the Model

#### Option A: REST API (Recommended for Production Use)

**Terminal 1** - Start the API server:
```bash
poetry run uvicorn src.score_pd.api:app --reload --port 8000
```

**Terminal 2** - Test the API:
```bash
poetry run python scripts/test_api.py
```

#### Option B: Python Script

```python
from src.score_pd.inference import CreditScorer

# Initialize scorer
scorer = CreditScorer(model_dir="models")

# Score an applicant
result = scorer.score_applicant({
    'fico_mid': 720,
    'dti': 25.5,
    'home_ownership': 'MORTGAGE',
    'verification_status': 'Verified',
    'inq_last_6mths_bin': '1',
    'emp_length_yrs': 5,
    # ... other features
})

print(f"Decision: {result['decision']}")
print(f"Default Probability: {result['default_probability']:.4f}")
print(f"Confidence Score: {result['confidence_score']:.4f}")
```

## Repository Structure

```
credit-scoring/
├── pyproject.toml              # Poetry dependencies and project metadata
├── poetry.lock                 # Locked dependency versions
├── README.md                   # This file
│
├── config/                     # Configuration files
│   └── config.yaml
│
├── models/                     # Trained model artifacts
│   ├── catboost_model.cbm      # Selected model (CatBoost)
│   ├── preprocessor.pkl        # Feature preprocessing pipeline
│   ├── calibrator_cat.pkl      # Probability calibration
│   ├── feature_list.txt        # Required feature names
│   ├── threshold_analysis.csv  # Business threshold analysis
│   ├── shap_importance.csv     # Feature importance scores
│   └── ks_analysis_validation.csv  # Model performance metrics
│
├── nb/                         # Jupyter notebooks (analysis pipeline)
│   ├── 1.Data_labeling.ipynb           # Target definition & credit lifetime
│   ├── 2.EDA_selected_sample.ipynb     # EDA & feature engineering
│   ├── 3.Model_Development.ipynb       # Model training & evaluation
│   └── functions.py                     # Helper functions for notebooks
│
├── src/                        # Source code
│   └── score_pd/               # Main package
│       ├── __init__.py
│       ├── functions.py        # Utility classes (DataProfile, ModelEvaluation, etc.)
│       ├── inference.py        # Model inference module
│       └── api.py              # FastAPI REST API
│
├── scripts/                    # Utility scripts
│   ├── test_api.py            # API testing script
│   └── example_request.json   # Sample API request
│
├── data/                       # Data files (not in repo)
│   ├── raw/
│   └── processed/
│
└── reports/                    # Generated reports and visualizations
```

## Notebooks

The analysis is organized in 3 sequential notebooks:

### 1. Data Labeling (`1.Data_labeling.ipynb`)
- **Purpose**: Define target variable and establish data quality baseline

### 2. Exploratory Data Analysis (`2.EDA_selected_sample.ipynb`)
- **Purpose**: Feature engineering and selection

### 3. Model Development (`3.Model_Development.ipynb`)
- **Purpose**: Model training, evaluation, and selection

## Testing the API

### Health Check

```bash
curl http://localhost:8000/health
```

### Score a Single Application

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "fico_mid": 720,
    "dti": 25.5,
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "inq_last_6mths_bin": "1",
    "emp_length_yrs": 5
  }'
```

### Expected Response

```json
{
  "application_id": 0,
  "decision": "APPROVED",
  "default_probability": 0.0847,
  "confidence_score": 0.8234,
  "timestamp": "2025-10-09T15:30:00"
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service health check |
| `/health` | GET | Detailed system status |
| `/score` | POST | Score single application |
| `/score/batch` | POST | Score multiple applications |
| `/model/info` | GET | Model metadata and configuration |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

## Dependencies

Main packages (managed by Poetry):

**ML & Data Science:**
- pandas, numpy, scikit-learn
- catboost, xgboost, lightgbm
- shap (model interpretability)

**API & Production:**
- fastapi, uvicorn, pydantic
- joblib (model serialization)

**Visualization:**
- matplotlib, seaborn

**Development:**
- jupyter, pytest

See `pyproject.toml` for complete dependency list with versions.

## Production Deployment (Conceptual)

### Proposed AWS Architecture

```
Application → API Gateway → Lambda (preprocessing)
                                ↓
                      SageMaker Endpoint (CatBoost)
                                ↓
                          Response + Logging
```

## Troubleshooting

### Poetry not found
```bash
# Add Poetry to PATH (macOS/Linux)
export PATH="$HOME/.local/bin:$PATH"

# Restart terminal and verify
poetry --version
```

### Module import errors
```bash
# Ensure you're in poetry shell
poetry shell

# Or prefix commands with poetry run
poetry run python your_script.py
```

### Port already in use
```bash
# Use different port
poetry run uvicorn src.score_pd.api:app --reload --port 8001
```

### Wrong Python version
```bash
# Check current version
python --version

# Use pyenv to switch
pyenv local 3.11.6
poetry env use python3.11
poetry install
```

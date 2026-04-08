# Mental Health Risk Prediction System

Production-ready MLOps project to predict mental health risk level (`Low`, `Medium`, `High`) from lifestyle input data.

## 1. Project Structure

```text
mlops-mental-health/
├── data/
│   └── mental_health.csv
├── src/
│   ├── preprocess.py
│   └── train.py
├── model/
│   ├── model.pkl                # generated after training
│   ├── label_encoder.pkl        # generated after training
│   └── metrics.json             # generated after training
├── api/
│   └── main.py
├── .github/workflows/deploy.yml
├── requirements.txt
├── Dockerfile
└── README.md
```

## 2. Dataset Format

Place your CSV file at `data/mental_health.csv`.

Required columns:
- `sleep_hours`
- `stress_level`
- `work_hours`
- `social_activity`
- `physical_activity`
- `screen_time`
- `risk_level` (target: `Low`, `Medium`, `High`)

## 3. Setup and Installation

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Train the Model

```bash
python -m src.train --data-path data/mental_health.csv --output-dir model --experiment-name mental-health-risk-prediction
```

What training does:
- Loads and validates data
- Handles missing values (median imputation)
- Encodes labels (`Low`, `Medium`, `High`)
- Scales numeric features
- If `risk_level` is missing, derives labels using explicit lifestyle rules:
  - `High`: high stress with poor sleep and/or other severe risk signals
  - `Low`: healthy sleep, low stress, balanced work/social/physical/screen profile
  - `Medium`: moderate or mixed lifestyle profile
- Trains `RandomForestClassifier`
- Runs basic hyperparameter tuning with `GridSearchCV`
- Logs params/metrics/model to MLflow
- Saves:
  - `model/model.pkl`
  - `model/label_encoder.pkl`
  - `model/metrics.json`

## 5. Run FastAPI Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /` health check
- `POST /predict` inference

Browser UI:
- `GET /ui` opens the interactive prediction dashboard
- `GET /` also serves the UI when opened in a browser, while still returning the JSON health message for API clients

## 6. Sample Prediction Request

### JSON Input

```json
{
  "sleep_hours": 6.5,
  "stress_level": 8,
  "work_hours": 9.0,
  "social_activity": 1.5,
  "physical_activity": 2.0,
  "screen_time": 7.5
}
```

### cURL Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sleep_hours": 6.5,
    "stress_level": 8,
    "work_hours": 9.0,
    "social_activity": 1.5,
    "physical_activity": 2.0,
    "screen_time": 7.5
  }'
```

Expected response:

```json
{
  "risk_level": "Medium"
}
```

## 7. MLflow Tracking

MLflow run artifacts are stored locally in `mlruns/` by default.

To view tracking UI:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000`.

## 8. Docker Run

Build image:

```bash
docker build -t mental-health-api .
```

Run container:

```bash
docker run -p 8000:8000 mental-health-api
```

## 9. Azure App Service Compatibility

- Python 3.10 compatible
- Lightweight model (`RandomForestClassifier`)
- No heavy deep learning dependencies
- Startup command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 10. GitHub Actions Azure Deployment

Workflow file: `.github/workflows/deploy.yml`

What the workflow does on push to `main` or `master`:
- Installs dependencies
- Runs Python syntax checks
- Trains model in CI and generates artifacts in `model/`
- Uploads the app package
- Deploys to Azure App Service

### Required GitHub repository secrets

- `AZURE_WEBAPP_NAME`: your Azure App Service name
- `AZURE_WEBAPP_PUBLISH_PROFILE`: publish profile XML content downloaded from Azure

### Azure App Service runtime settings

- OS: Linux
- Runtime: Python 3.10
- Startup command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Deployment data requirement

The workflow expects one of these files in the repository:
- `data/mental_health.csv` (preferred)
- `Mental_Health_Lifestyle_Dataset.csv`

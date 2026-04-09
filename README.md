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

Note: the Docker image and local runtime should use `scikit-learn==1.7.1` to match the saved model artifacts and avoid version-mismatch warnings during unpickling.

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
- `GET /` interactive dashboard UI
- `GET /health` API and model health status
- `POST /predict` enhanced inference
- `GET /history` recent prediction history (SQLite)
- `GET /history/stats` summary analytics

Dashboard features:
- Professional responsive UI with result cards and risk meter
- Color-coded levels (`Low` green, `Medium` yellow, `High` red)
- Explanation and key influencing factors
- Personalized recommendations
- Input radar chart and feature-importance chart (Chart.js)
- Local history + trend chart from SQLite tracking

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
  "risk_score": 78.4,
  "risk_level": "High",
  "explanation": "Predicted High risk with score 78.4%. Main signals come from lifestyle patterns such as stress, sleep, activity, and screen exposure.",
  "key_factors": [
    {
      "factor": "stress_level",
      "impact": "high",
      "reason": "Stress is very high."
    }
  ],
  "recommendations": [
    "Improve sleep consistency: target 7-9 hours with fixed bed/wake times."
  ],
  "probabilities": {
    "Low": 0.05,
    "Medium": 0.31,
    "High": 0.64
  },
  "feature_importance": [
    {
      "feature": "stress_level",
      "importance": 0.2891
    }
  ]
}
```

## 6. Prediction History Storage

Predictions are stored in SQLite automatically at:

`data/predictions.db`

This enables:
- session-level tracking
- history table visualization in the dashboard
- risk score trend chart

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
docker run -p 8000:8000 -v "${PWD}/data:/app/data" mental-health-api
```

Then open:
- `http://127.0.0.1:8000/` for the dashboard UI with charts and recommendations
- `http://127.0.0.1:8000/docs` for API docs
- `http://127.0.0.1:8000/history` for prediction history JSON

The dashboard includes the risk meter, recommendation cards, and Chart.js visualizations directly inside the Docker container.

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

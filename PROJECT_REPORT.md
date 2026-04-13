# Mental Health Risk Prediction System - Project Report

## 1. Project Overview

### 1.1 Title
Mental Health Risk Prediction System

### 1.2 Objective
The objective of this project is to predict mental health risk level (Low, Medium, High) from lifestyle-related inputs using a machine learning model and provide an interactive web dashboard for end users.

### 1.3 Problem Statement
Mental health risk is influenced by daily patterns such as sleep quality, stress level, work pressure, social engagement, physical activity, and screen exposure. This project converts these signals into a risk prediction and actionable recommendations.

---

## 2. Tech Stack

### 2.1 Programming Language
- Python

### 2.2 Machine Learning
- scikit-learn (RandomForestClassifier, preprocessing pipeline, model selection)
- pandas, numpy
- joblib (model persistence)
- MLflow (experiment tracking)

### 2.3 Backend
- FastAPI
- Pydantic
- Uvicorn / Gunicorn

### 2.4 Frontend
- HTML, CSS, JavaScript (served by FastAPI)
- Chart.js for visualizations

### 2.5 Data Storage
- SQLite for prediction history (`data/predictions.db`)

### 2.6 Deployment / DevOps
- Docker
- GitHub Actions CI/CD
- Azure App Service (Linux)

---

## 3. Project Structure

```
mlops project/
├── api/
│   └── main.py
├── src/
│   ├── preprocess.py
│   └── train.py
├── model/
│   ├── model.pkl
│   ├── label_encoder.pkl
│   └── metrics.json
├── data/
│   ├── predictions.db
│   └── mental_health.csv (or dataset file)
├── mlruns/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 4. Dataset and Features

### 4.1 Input Features
- sleep_hours
- stress_level
- work_hours
- social_activity
- physical_activity
- screen_time

### 4.2 Target
- risk_level: Low / Medium / High

### 4.3 Column Alias Handling
The preprocessing module supports alternate column names (for portability across datasets) and maps them into canonical feature names.

---

## 5. Preprocessing Pipeline

Implemented in `src/preprocess.py`.

### 5.1 Data Loading
- CSV loaded from configurable path

### 5.2 Feature Standardization
- Column mapping from aliases to canonical names
- Conversion of categorical-like entries (stress/exercise) into numeric values

### 5.3 Missing Value Handling
- Numeric conversion with coercion
- Median imputation through `SimpleImputer(strategy="median")`

### 5.4 Label Handling
- If target exists: normalize labels to Low/Medium/High
- If target missing: derive labels via rule-based logic from lifestyle signals

### 5.5 Scaling and Split
- Standard scaling using `StandardScaler`
- Train/test split with stratification

### 5.6 Encoding
- Label encoding for target class values

---

## 6. Model Training

Implemented in `src/train.py`.

### 6.1 Model Choice
- RandomForestClassifier

### 6.2 Pipeline
- End-to-end sklearn `Pipeline`:
  - preprocessor
  - classifier

### 6.3 Hyperparameter Tuning
- `GridSearchCV` for model selection
- Tuned parameters include:
  - n_estimators
  - max_depth
  - min_samples_split
  - min_samples_leaf

### 6.4 Evaluation
- Accuracy
- Confusion matrix
- Classification report (precision, recall, F1)

### 6.5 Artifacts Saved
- `model/model.pkl`
- `model/label_encoder.pkl`
- `model/metrics.json`

### 6.6 MLflow Tracking
- Logs parameters, metrics, and model artifact into `mlruns/`

---

## 7. Backend API (FastAPI)

Implemented in `api/main.py` with `app.py` as entrypoint import wrapper.

### 7.1 Main Endpoints
- `GET /` -> interactive dashboard UI
- `GET /health` -> backend health and model status
- `POST /predict` -> returns risk prediction and explainability data
- `GET /history` -> recent predictions from SQLite
- `GET /history/stats` -> aggregate statistics from history

### 7.2 Predict Response Fields
- risk_score
- risk_level
- explanation
- key_factors
- recommendations
- probabilities
- feature_importance
- persistence_status
- prediction_mode

### 7.3 Robust Inference Mode
To avoid runtime failures caused by model serialization/version mismatch, the backend includes:
- model-based inference path
- rule-based fallback inference path

This ensures user-facing predictions are returned even if pickle runtime compatibility issues occur.

---

## 8. Frontend Dashboard

Served directly from FastAPI (`api/main.py`) as embedded HTML/CSS/JS.

### 8.1 UI Components
- Input form for all features
- Prediction result card
- Risk badge and risk score meter
- Explanation area
- Recommendations list
- Key factor chips

### 8.2 Visualizations
- Radar chart for input profile
- Bar chart for feature importance
- Line chart for risk score trend from history

### 8.3 UX Features
- Responsive layout
- Color-coded risk status:
  - Green: Low
  - Yellow: Medium
  - Red: High

---

## 9. Database Layer

### 9.1 Engine
- SQLite

### 9.2 File
- `data/predictions.db`

### 9.3 Stored Fields
- Timestamp
- Inputs
- Risk score/level
- Explanation
- Recommendations

### 9.4 Purpose
- History table in dashboard
- Trend graph
- Summary statistics for tracking

---

## 10. Dockerization

### 10.1 Dockerfile Highlights
- Base image: `python:3.10-slim`
- Installs dependencies from `requirements.txt`
- Sets environment variables for runtime
- Exposes port `8000`
- Starts app via Gunicorn with Uvicorn worker
- Defines writable data volume for SQLite history

### 10.2 Important Runtime Setting
- `PREDICTIONS_DB_PATH=/app/data/predictions.db`

### 10.3 Run Example
```bash
docker build -t mental-health-api .
docker run -p 8000:8000 -v "${PWD}/data:/app/data" mental-health-api
```

---

## 11. CI/CD Pipeline

Implemented in `.github/workflows/deploy.yml`.

### 11.1 CI Steps
- Checkout code
- Setup Python
- Install dependencies
- Syntax checks
- Model training in workflow
- Artifact verification
- Package upload

### 11.2 CD Steps
- Download build artifact
- Deploy to Azure Web App

### 11.3 Trigger
- Push to main/master
- Manual workflow dispatch

---

## 12. Deployment Target

### 12.1 Platform
- Azure App Service (Linux)

### 12.2 Startup
- Gunicorn with Uvicorn worker

### 12.3 Environment Considerations
- Writable path handling for SQLite
- Resilience against model/runtime version mismatch

---

## 13. Dependency Summary

From `requirements.txt`:
- pandas
- numpy
- scikit-learn (pinned)
- fastapi
- uvicorn
- gunicorn
- joblib
- mlflow

---

## 14. Key Outcomes

- End-to-end ML pipeline from preprocessing to deployment
- Explainable risk prediction output for users
- Production-style API and dashboard integration
- Persistent tracking and visualization of prediction history
- Containerized runtime + CI/CD automation

---

## 15. Future Enhancements

- User authentication and role-based dashboard access
- Multi-user profile tracking
- Time-series analytics and longitudinal risk insights
- Periodic model retraining automation
- Explainability expansion (SHAP/LIME)
- Alerting and wellness intervention workflows

---

## 16. Conclusion

This project demonstrates a complete MLOps lifecycle: data preprocessing, model training, experiment tracking, API serving, interactive analytics dashboard, persistence, containerization, and cloud deployment automation. It is suitable for academic presentation, portfolio demonstration, and practical extension into a larger health analytics platform.

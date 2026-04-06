from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
ENCODER_PATH = PROJECT_ROOT / "model" / "label_encoder.pkl"

app = FastAPI(title="Mental Health Risk Prediction API", version="1.0.0")


class RiskInput(BaseModel):
    sleep_hours: float = Field(..., ge=0, le=24)
    stress_level: int = Field(..., ge=0, le=10)
    work_hours: float = Field(..., ge=0, le=24)
    social_activity: float = Field(..., ge=0)
    physical_activity: float = Field(..., ge=0)
    screen_time: float = Field(..., ge=0, le=24)


model = None
label_encoder = None


@app.on_event("startup")
def load_artifacts() -> None:
    global model, label_encoder

    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        # API can start and expose health endpoint before model is trained.
        model = None
        label_encoder = None
        return

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)


@app.get("/")
def health_check() -> dict:
    return {"message": "Mental Health API running"}


@app.post("/predict")
def predict(payload: RiskInput) -> dict:
    if model is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the model first using src/train.py.",
        )

    input_df = pd.DataFrame([payload.model_dump()])

    try:
        prediction_encoded = model.predict(input_df)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        return {"risk_level": str(prediction_label)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

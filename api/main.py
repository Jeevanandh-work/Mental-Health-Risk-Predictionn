from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


UI_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Mental Health Risk Predictor</title>
    <style>
        :root {
            --bg: #f2efe9;
            --ink: #1c2b2a;
            --muted: #4f5f5d;
            --panel: #fffdf8;
            --line: #d8d2c6;
            --accent: #0f766e;
            --accent-2: #f59e0b;
            --danger: #b91c1c;
            --ok: #166534;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            color: var(--ink);
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            background:
                radial-gradient(circle at 20% 0%, rgba(15,118,110,0.12), transparent 35%),
                radial-gradient(circle at 90% 20%, rgba(245,158,11,0.14), transparent 35%),
                var(--bg);
            min-height: 100vh;
        }

        .wrap {
            max-width: 980px;
            margin: 0 auto;
            padding: 24px 16px 40px;
        }

        .hero {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }

        h1 {
            margin: 0;
            font-size: clamp(1.8rem, 4vw, 2.8rem);
            letter-spacing: -0.02em;
        }

        p { color: var(--muted); }

        .grid {
            margin-top: 18px;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        label {
            font-weight: 600;
            font-size: 0.92rem;
        }

        input {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 11px;
            font-size: 0.98rem;
            background: #fff;
            color: var(--ink);
        }

        .actions {
            margin-top: 16px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        button {
            border: none;
            border-radius: 12px;
            padding: 11px 14px;
            font-weight: 700;
            cursor: pointer;
        }

        .primary {
            background: var(--accent);
            color: #fff;
        }

        .secondary {
            background: #ece8df;
            color: var(--ink);
        }

        .result {
            margin-top: 18px;
            border: 1px solid var(--line);
            background: #fff;
            border-radius: 14px;
            padding: 14px;
            min-height: 90px;
        }

        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 7px 11px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .pill.low { background: #dcfce7; color: var(--ok); }
        .pill.medium { background: #fef3c7; color: #92400e; }
        .pill.high { background: #fee2e2; color: var(--danger); }

        @media (max-width: 760px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <main class="wrap">
        <section class="hero">
            <h1>Mental Health Risk Predictor</h1>
            <p>Fill in daily lifestyle patterns to predict risk level as Low, Medium, or High.</p>

            <div class="grid">
                <div class="field"><label for="sleep_hours">Sleep Hours</label><input id="sleep_hours" type="number" step="0.1" min="0" max="24" value="6.5"></div>
                <div class="field"><label for="stress_level">Stress Level (0-10)</label><input id="stress_level" type="number" step="1" min="0" max="10" value="8"></div>
                <div class="field"><label for="work_hours">Work Hours</label><input id="work_hours" type="number" step="0.1" min="0" max="24" value="9"></div>
                <div class="field"><label for="social_activity">Social Activity</label><input id="social_activity" type="number" step="0.1" min="0" value="1.5"></div>
                <div class="field"><label for="physical_activity">Physical Activity</label><input id="physical_activity" type="number" step="0.1" min="0" value="2"></div>
                <div class="field"><label for="screen_time">Screen Time</label><input id="screen_time" type="number" step="0.1" min="0" max="24" value="7.5"></div>
            </div>

            <div class="actions">
                <button class="primary" onclick="predict()">Predict</button>
                <button class="secondary" onclick="resetValues()">Load Sample</button>
            </div>

            <div class="result" id="result">Prediction output will appear here.</div>
        </section>
    </main>

    <script>
        function payload() {
            return {
                sleep_hours: Number(document.getElementById('sleep_hours').value),
                stress_level: Number(document.getElementById('stress_level').value),
                work_hours: Number(document.getElementById('work_hours').value),
                social_activity: Number(document.getElementById('social_activity').value),
                physical_activity: Number(document.getElementById('physical_activity').value),
                screen_time: Number(document.getElementById('screen_time').value)
            };
        }

        function resetValues() {
            document.getElementById('sleep_hours').value = 6.5;
            document.getElementById('stress_level').value = 8;
            document.getElementById('work_hours').value = 9;
            document.getElementById('social_activity').value = 1.5;
            document.getElementById('physical_activity').value = 2;
            document.getElementById('screen_time').value = 7.5;
            document.getElementById('result').textContent = 'Sample values loaded.';
        }

        function riskClass(level) {
            const value = String(level || '').toLowerCase();
            if (value.includes('high')) return 'high';
            if (value.includes('medium')) return 'medium';
            return 'low';
        }

        async function predict() {
            const result = document.getElementById('result');
            result.textContent = 'Predicting...';
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload())
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || 'Prediction failed');
                }
                const level = data.risk_level;
                result.innerHTML = '<div class="pill ' + riskClass(level) + '">' + level + ' Risk</div><div>Prediction completed successfully.</div>';
            } catch (error) {
                result.innerHTML = '<div class="pill high">Error</div><div>' + error.message + '</div>';
            }
        }
    </script>
</body>
</html>
""".strip()


@app.on_event("startup")
def load_artifacts() -> None:
        global model, label_encoder

        if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
                model = None
                label_encoder = None
                return

        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
        return UI_HTML


@app.get("/health")
def health() -> dict:
        loaded = model is not None and label_encoder is not None
        return {
                "status": "ok",
                "model_loaded": loaded,
        }


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

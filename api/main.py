from __future__ import annotations

from textwrap import dedent
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


UI_HTML = dedent(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Mental Health Risk Prediction</title>
            <style>
                :root {
                    --bg: #07111f;
                    --panel: rgba(12, 21, 39, 0.82);
                    --panel-strong: #0f1c33;
                    --text: #eaf2ff;
                    --muted: #9cb2d8;
                    --accent: #66e3c4;
                    --accent-2: #7ab8ff;
                    --danger: #ff6b8a;
                    --warning: #ffd36b;
                    --success: #76f2a8;
                    --border: rgba(130, 162, 219, 0.18);
                    --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
                }

                * { box-sizing: border-box; }

                body {
                    margin: 0;
                    min-height: 100vh;
                    font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
                    color: var(--text);
                    background:
                        radial-gradient(circle at top left, rgba(102, 227, 196, 0.18), transparent 28%),
                        radial-gradient(circle at top right, rgba(122, 184, 255, 0.2), transparent 26%),
                        linear-gradient(145deg, #050b14 0%, #081121 48%, #0a1630 100%);
                }

                .wrap {
                    width: min(1180px, calc(100% - 32px));
                    margin: 0 auto;
                    padding: 32px 0 56px;
                }

                .hero {
                    display: grid;
                    grid-template-columns: 1.35fr 0.95fr;
                    gap: 24px;
                    align-items: stretch;
                }

                .card {
                    background: var(--panel);
                    border: 1px solid var(--border);
                    backdrop-filter: blur(18px);
                    box-shadow: var(--shadow);
                    border-radius: 24px;
                    overflow: hidden;
                }

                .hero-main {
                    padding: 34px;
                    position: relative;
                }

                .eyebrow {
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 12px;
                    border-radius: 999px;
                    background: rgba(102, 227, 196, 0.12);
                    color: var(--accent);
                    font-size: 13px;
                    letter-spacing: 0.02em;
                    margin-bottom: 18px;
                }

                h1 {
                    margin: 0;
                    font-size: clamp(2.2rem, 5vw, 4.2rem);
                    line-height: 0.95;
                    letter-spacing: -0.04em;
                }

                .sub {
                    margin: 18px 0 0;
                    max-width: 58ch;
                    color: var(--muted);
                    font-size: 1rem;
                    line-height: 1.75;
                }

                .stats {
                    display: grid;
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                    gap: 14px;
                    margin-top: 26px;
                }

                .stat {
                    padding: 16px;
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 18px;
                }

                .stat strong {
                    display: block;
                    font-size: 1.1rem;
                    margin-bottom: 6px;
                }

                .stat span {
                    color: var(--muted);
                    font-size: 0.92rem;
                }

                .side {
                    padding: 26px;
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }

                .side h2 {
                    margin: 0 0 6px;
                    font-size: 1.1rem;
                }

                .pill-list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }

                .pill {
                    padding: 10px 12px;
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.07);
                    color: var(--text);
                    font-size: 13px;
                }

                .section {
                    margin-top: 24px;
                    display: grid;
                    grid-template-columns: 1.2fr 0.8fr;
                    gap: 24px;
                }

                .form-card, .result-card {
                    padding: 26px;
                }

                .form-grid {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 16px;
                }

                .field {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }

                .field label {
                    font-size: 0.9rem;
                    color: var(--muted);
                }

                .field input {
                    width: 100%;
                    padding: 14px 14px;
                    border-radius: 14px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    outline: none;
                    background: rgba(8, 15, 28, 0.85);
                    color: var(--text);
                    font-size: 15px;
                    transition: border-color 0.2s ease, transform 0.2s ease;
                }

                .field input:focus {
                    border-color: rgba(102, 227, 196, 0.65);
                    transform: translateY(-1px);
                }

                .actions {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 12px;
                    margin-top: 18px;
                }

                button {
                    border: none;
                    border-radius: 14px;
                    padding: 13px 18px;
                    font-weight: 700;
                    cursor: pointer;
                    transition: transform 0.18s ease, opacity 0.18s ease;
                }

                button:hover { transform: translateY(-1px); }

                .primary {
                    background: linear-gradient(135deg, var(--accent), var(--accent-2));
                    color: #06101d;
                }

                .secondary {
                    background: rgba(255, 255, 255, 0.07);
                    color: var(--text);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                }

                .result-box {
                    margin-top: 10px;
                    padding: 18px;
                    border-radius: 18px;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    background: rgba(255, 255, 255, 0.04);
                    min-height: 168px;
                }

                .risk {
                    display: inline-flex;
                    align-items: center;
                    gap: 10px;
                    padding: 10px 14px;
                    border-radius: 999px;
                    font-weight: 800;
                    letter-spacing: 0.02em;
                    margin-bottom: 10px;
                }

                .risk.low { background: rgba(118, 242, 168, 0.12); color: var(--success); }
                .risk.medium { background: rgba(255, 211, 107, 0.14); color: var(--warning); }
                .risk.high { background: rgba(255, 107, 138, 0.14); color: var(--danger); }

                .hint, .footer-note {
                    color: var(--muted);
                    line-height: 1.7;
                    font-size: 0.95rem;
                }

                .footer-note {
                    margin-top: 14px;
                }

                .tiny {
                    color: var(--muted);
                    font-size: 0.84rem;
                    margin-top: 8px;
                }

                .status {
                    margin-top: 14px;
                    font-weight: 700;
                }

                @media (max-width: 980px) {
                    .hero, .section { grid-template-columns: 1fr; }
                }

                @media (max-width: 640px) {
                    .wrap { width: min(100% - 20px, 100%); padding-top: 16px; }
                    .hero-main, .side, .form-card, .result-card { padding: 20px; }
                    .stats, .form-grid { grid-template-columns: 1fr; }
                }
            </style>
        </head>
        <body>
            <main class="wrap">
                <section class="hero">
                    <div class="card hero-main">
                        <div class="eyebrow">Mental Health Risk Prediction System</div>
                        <h1>Estimate risk levels from lifestyle patterns.</h1>
                        <p class="sub">
                            Enter sleep, stress, work, social, physical, and screen-time values to get a
                            Low, Medium, or High risk prediction from the trained RandomForest model.
                        </p>
                        <div class="stats">
                            <div class="stat"><strong>FastAPI</strong><span>Clean backend with JSON prediction API</span></div>
                            <div class="stat"><strong>MLflow</strong><span>Training tracked and reproducible</span></div>
                            <div class="stat"><strong>Azure-ready</strong><span>Lightweight deployment for App Service</span></div>
                        </div>
                    </div>
                    <aside class="card side">
                        <div>
                            <h2>Supported inputs</h2>
                            <p class="hint">All values map directly to the trained model features.</p>
                        </div>
                        <div class="pill-list">
                            <span class="pill">sleep_hours</span>
                            <span class="pill">stress_level</span>
                            <span class="pill">work_hours</span>
                            <span class="pill">social_activity</span>
                            <span class="pill">physical_activity</span>
                            <span class="pill">screen_time</span>
                        </div>
                        <div>
                            <h2>Try the API</h2>
                            <p class="hint">Use the form below or call <code>/predict</code> directly from your app.</p>
                        </div>
                    </aside>
                </section>

                <section class="section">
                    <div class="card form-card">
                        <h2 style="margin-top: 0;">Lifestyle inputs</h2>
                        <div class="form-grid">
                            <div class="field"><label for="sleep_hours">Sleep Hours</label><input id="sleep_hours" type="number" min="0" max="24" step="0.1" value="6.5" /></div>
                            <div class="field"><label for="stress_level">Stress Level</label><input id="stress_level" type="number" min="0" max="10" step="1" value="8" /></div>
                            <div class="field"><label for="work_hours">Work Hours</label><input id="work_hours" type="number" min="0" max="24" step="0.1" value="9" /></div>
                            <div class="field"><label for="social_activity">Social Activity</label><input id="social_activity" type="number" min="0" step="0.1" value="1.5" /></div>
                            <div class="field"><label for="physical_activity">Physical Activity</label><input id="physical_activity" type="number" min="0" step="0.1" value="2" /></div>
                            <div class="field"><label for="screen_time">Screen Time</label><input id="screen_time" type="number" min="0" max="24" step="0.1" value="7.5" /></div>
                        </div>
                        <div class="actions">
                            <button class="primary" onclick="predictRisk()">Predict Risk</button>
                            <button class="secondary" onclick="loadSample()">Load Sample</button>
                            <button class="secondary" onclick="clearOutput()">Clear Result</button>
                        </div>
                        <div class="tiny">Tip: High stress + low sleep usually leads to a higher risk result.</div>
                        <div id="status" class="status"></div>
                    </div>

                    <div class="card result-card">
                        <h2 style="margin-top: 0;">Prediction result</h2>
                        <div id="resultBox" class="result-box">
                            <div class="hint">Your prediction will appear here after you submit the form.</div>
                        </div>
                        <p class="footer-note">
                            This UI calls the same FastAPI <code>/predict</code> endpoint used by the deployment pipeline.
                        </p>
                    </div>
                </section>
            </main>

            <script>
                const sample = {
                    sleep_hours: 6.5,
                    stress_level: 8,
                    work_hours: 9,
                    social_activity: 1.5,
                    physical_activity: 2,
                    screen_time: 7.5,
                };

                function readInputs() {
                    return {
                        sleep_hours: Number(document.getElementById('sleep_hours').value),
                        stress_level: Number(document.getElementById('stress_level').value),
                        work_hours: Number(document.getElementById('work_hours').value),
                        social_activity: Number(document.getElementById('social_activity').value),
                        physical_activity: Number(document.getElementById('physical_activity').value),
                        screen_time: Number(document.getElementById('screen_time').value),
                    };
                }

                function loadSample() {
                    Object.entries(sample).forEach(([key, value]) => {
                        document.getElementById(key).value = value;
                    });
                    document.getElementById('status').textContent = 'Sample values loaded.';
                }

                function clearOutput() {
                    document.getElementById('resultBox').innerHTML = '<div class="hint">Your prediction will appear here after you submit the form.</div>';
                    document.getElementById('status').textContent = '';
                }

                function riskClass(level) {
                    const normalized = String(level || '').toLowerCase();
                    if (normalized.includes('high')) return 'high';
                    if (normalized.includes('medium')) return 'medium';
                    return 'low';
                }

                async function predictRisk() {
                    const payload = readInputs();
                    const status = document.getElementById('status');
                    const resultBox = document.getElementById('resultBox');
                    status.textContent = 'Predicting...';

                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload),
                        });

                        const data = await response.json();

                        if (!response.ok) {
                            throw new Error(data.detail || 'Prediction failed');
                        }

                        const level = data.risk_level;
                        resultBox.innerHTML = `
                            <div class="risk ${riskClass(level)}">${level} Risk</div>
                            <div class="hint">Prediction completed successfully. This result comes from the trained model and label encoder.</div>
                            <div class="tiny">You can change the values and predict again as many times as you want.</div>
                        `;
                        status.textContent = 'Prediction complete.';
                    } catch (error) {
                        resultBox.innerHTML = `<div class="risk high">Error</div><div class="hint">${error.message}</div>`;
                        status.textContent = 'Prediction error.';
                    }
                }
            </script>
        </body>
        </html>
        """
).strip()


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
def home():
    return {"message": "API running successfully"}


@app.get("/ui", response_class=HTMLResponse)
def ui_page() -> str:
    return UI_HTML


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

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
ENCODER_PATH = PROJECT_ROOT / "model" / "label_encoder.pkl"


def resolve_db_path() -> Path:
  # Azure App Service often mounts app code as read-only, so default to /home/data there.
  explicit = os.getenv("PREDICTIONS_DB_PATH")
  if explicit:
    return Path(explicit)

  if os.getenv("WEBSITE_INSTANCE_ID"):
    return Path("/home/data/predictions.db")

  return PROJECT_ROOT / "data" / "predictions.db"


DB_PATH = resolve_db_path()

FEATURE_COLUMNS = [
    "sleep_hours",
    "stress_level",
    "work_hours",
    "social_activity",
    "physical_activity",
    "screen_time",
]

SCORE_WEIGHTS = {
    "Low": 20,
    "Medium": 60,
    "High": 90,
}

app = FastAPI(title="Mental Health Risk Prediction API", version="2.0.0")


class RiskInput(BaseModel):
    sleep_hours: float = Field(..., ge=0, le=24)
    stress_level: int = Field(..., ge=0, le=10)
    work_hours: float = Field(..., ge=0, le=24)
    social_activity: float = Field(..., ge=0)
    physical_activity: float = Field(..., ge=0)
    screen_time: float = Field(..., ge=0, le=24)


model: Any = None
label_encoder: Any = None


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                sleep_hours REAL NOT NULL,
                stress_level INTEGER NOT NULL,
                work_hours REAL NOT NULL,
                social_activity REAL NOT NULL,
                physical_activity REAL NOT NULL,
                screen_time REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                explanation TEXT NOT NULL,
                recommendations_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def compute_risk_score(probabilities: dict[str, float]) -> float:
    weighted_score = 0.0
    for level, weight in SCORE_WEIGHTS.items():
        weighted_score += probabilities.get(level, 0.0) * float(weight)
    return max(0.0, min(100.0, weighted_score))


def get_feature_importance() -> list[dict[str, Any]]:
    if model is None:
        return []

    # Model is expected to be a sklearn pipeline with preprocessor + classifier.
    classifier = None
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")

    if classifier is None or not hasattr(classifier, "feature_importances_"):
        return []

    importances = classifier.feature_importances_
    if len(importances) != len(FEATURE_COLUMNS):
        return []

    rows = []
    for feature, importance in zip(FEATURE_COLUMNS, importances):
        rows.append({"feature": feature, "importance": round(float(importance), 4)})

    rows.sort(key=lambda x: x["importance"], reverse=True)
    return rows


def build_explanation(payload: RiskInput, risk_level: str, risk_score: float) -> tuple[str, list[dict[str, str]]]:
    factors: list[dict[str, str]] = []

    if payload.stress_level >= 8:
        factors.append({"factor": "stress_level", "impact": "high", "reason": "Stress is very high."})
    elif payload.stress_level >= 6:
        factors.append({"factor": "stress_level", "impact": "medium", "reason": "Stress is above healthy range."})

    if payload.sleep_hours < 5.5:
        factors.append({"factor": "sleep_hours", "impact": "high", "reason": "Sleep is too low for recovery."})
    elif payload.sleep_hours < 6.5:
        factors.append({"factor": "sleep_hours", "impact": "medium", "reason": "Sleep is slightly below ideal."})

    if payload.screen_time > 7:
        factors.append({"factor": "screen_time", "impact": "medium", "reason": "Screen time is elevated."})

    if payload.physical_activity < 4:
        factors.append({"factor": "physical_activity", "impact": "medium", "reason": "Physical activity is low."})

    if payload.social_activity < 4:
        factors.append({"factor": "social_activity", "impact": "medium", "reason": "Social activity is limited."})

    if payload.work_hours > 10:
        factors.append({"factor": "work_hours", "impact": "medium", "reason": "Long work hours can increase stress."})

    if not factors:
        factors.append({
            "factor": "overall_balance",
            "impact": "positive",
            "reason": "Inputs look relatively balanced and protective.",
        })

    explanation = (
        f"Predicted {risk_level} risk with score {risk_score:.1f}%. "
        "Main signals come from lifestyle patterns such as stress, sleep, activity, and screen exposure."
    )

    return explanation, factors


def build_recommendations(payload: RiskInput, risk_level: str) -> list[str]:
    recommendations: list[str] = []

    if payload.sleep_hours < 7:
        recommendations.append("Improve sleep consistency: target 7-9 hours with fixed bed/wake times.")
    if payload.stress_level >= 6:
        recommendations.append("Add daily stress-regulation practice (10-15 min breathing, mindfulness, or journaling).")
    if payload.screen_time > 6:
        recommendations.append("Reduce evening screen time and use a 60-minute digital wind-down before sleep.")
    if payload.physical_activity < 5:
        recommendations.append("Increase physical activity: at least 30 minutes, 5 days/week.")
    if payload.social_activity < 5:
        recommendations.append("Schedule meaningful social interactions during the week.")
    if payload.work_hours > 9:
        recommendations.append("Set work boundaries and include short breaks every 60-90 minutes.")

    if not recommendations:
        recommendations.append("Maintain your current balanced routine and continue weekly self-checks.")

    if risk_level == "High":
        recommendations.append("If symptoms persist, consult a licensed mental health professional.")

    return recommendations


def save_prediction(payload: RiskInput, risk_score: float, risk_level: str, explanation: str, recommendations: list[str]) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at,
                sleep_hours,
                stress_level,
                work_hours,
                social_activity,
                physical_activity,
                screen_time,
                risk_score,
                risk_level,
                explanation,
                recommendations_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                payload.sleep_hours,
                payload.stress_level,
                payload.work_hours,
                payload.social_activity,
                payload.physical_activity,
                payload.screen_time,
                float(risk_score),
                risk_level,
                explanation,
                json.dumps(recommendations),
            ),
        )
        conn.commit()


def safe_save_prediction(payload: RiskInput, risk_score: float, risk_level: str, explanation: str, recommendations: list[str]) -> str:
    try:
        save_prediction(payload, risk_score, risk_level, explanation, recommendations)
        return "saved"
    except Exception:
        # Prediction should still succeed even if persistence fails.
        return "save_failed"


@app.on_event("startup")
def load_artifacts() -> None:
    global model, label_encoder
    init_db()

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
def health() -> dict[str, Any]:
    loaded = model is not None and label_encoder is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "db_path": str(DB_PATH),
    }


@app.post("/predict")
def predict(payload: RiskInput) -> dict[str, Any]:
    if model is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the model first using src/train.py.",
        )

    input_df = pd.DataFrame([payload.model_dump()])

    try:
        prediction_encoded = model.predict(input_df)
        predicted_label = str(label_encoder.inverse_transform(prediction_encoded)[0])

        probabilities: dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            classes = list(label_encoder.classes_)
            probabilities = {str(cls): float(prob) for cls, prob in zip(classes, proba)}

        risk_score = compute_risk_score(probabilities) if probabilities else float(SCORE_WEIGHTS.get(predicted_label, 50))

        explanation, key_factors = build_explanation(payload, predicted_label, risk_score)
        recommendations = build_recommendations(payload, predicted_label)
        feature_importance = get_feature_importance()

        persistence_status = safe_save_prediction(payload, risk_score, predicted_label, explanation, recommendations)

        return {
            "risk_score": round(risk_score, 1),
            "risk_level": predicted_label,
            "explanation": explanation,
            "key_factors": key_factors,
            "recommendations": recommendations,
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "feature_importance": feature_importance,
            "persistence_status": persistence_status,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.get("/history")
def get_history(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, sleep_hours, stress_level, work_hours,
                       social_activity, physical_activity, screen_time,
                       risk_score, risk_level, explanation, recommendations_json
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except Exception:
        return {"count": 0, "items": [], "status": "db_unavailable"}

    history = []
    for row in rows:
        history.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "inputs": {
                    "sleep_hours": row["sleep_hours"],
                    "stress_level": row["stress_level"],
                    "work_hours": row["work_hours"],
                    "social_activity": row["social_activity"],
                    "physical_activity": row["physical_activity"],
                    "screen_time": row["screen_time"],
                },
                "risk_score": row["risk_score"],
                "risk_level": row["risk_level"],
                "explanation": row["explanation"],
                "recommendations": json.loads(row["recommendations_json"]),
            }
        )

    return {"count": len(history), "items": history, "status": "ok"}


@app.get("/history/stats")
def get_history_stats() -> dict[str, Any]:
    try:
        with get_db_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
            avg_score_row = conn.execute("SELECT AVG(risk_score) as avg_score FROM predictions").fetchone()
            avg_score = float(avg_score_row["avg_score"]) if avg_score_row["avg_score"] is not None else 0.0

            level_rows = conn.execute(
                """
                SELECT risk_level, COUNT(*) as count
                FROM predictions
                GROUP BY risk_level
                """
            ).fetchall()
    except Exception:
        return {
            "total_predictions": 0,
            "average_risk_score": 0.0,
            "level_breakdown": {"Low": 0, "Medium": 0, "High": 0},
            "status": "db_unavailable",
        }

    level_breakdown = {"Low": 0, "Medium": 0, "High": 0}
    for row in level_rows:
        level_breakdown[str(row["risk_level"])] = int(row["count"])

    return {
        "total_predictions": int(total),
        "average_risk_score": round(avg_score, 2),
        "level_breakdown": level_breakdown,
        "status": "ok",
    }


UI_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mental Health Risk AI Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #f6f2e9;
      --panel: #fffdf8;
      --panel-2: #f3ede2;
      --ink: #1f2937;
      --muted: #5b6472;
      --line: #dfd7c8;
      --primary: #0f766e;
      --low: #15803d;
      --medium: #ca8a04;
      --high: #b91c1c;
      --shadow: 0 12px 36px rgba(31, 41, 55, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background:
        radial-gradient(circle at 10% 0%, rgba(15, 118, 110, 0.15), transparent 32%),
        radial-gradient(circle at 95% 20%, rgba(202, 138, 4, 0.16), transparent 28%),
        var(--bg);
      min-height: 100vh;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }

    .hero {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 22px;
      margin-bottom: 16px;
    }

    h1 {
      margin: 0;
      letter-spacing: -0.02em;
      font-size: clamp(1.8rem, 3vw, 2.6rem);
    }

    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.6;
    }

    .layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 16px;
    }

    .card h2 {
      margin-top: 0;
      margin-bottom: 10px;
      font-size: 1.15rem;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .field { display: flex; flex-direction: column; gap: 6px; }

    .field label {
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--muted);
    }

    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 11px;
      padding: 10px;
      font-size: 0.95rem;
      background: #fff;
      color: var(--ink);
    }

    .actions {
      margin-top: 14px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    button {
      border: none;
      border-radius: 10px;
      padding: 10px 12px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.2s ease;
    }

    button:hover { transform: translateY(-1px); }

    .primary { background: var(--primary); color: #fff; }
    .secondary { background: var(--panel-2); color: var(--ink); }

    .result-header {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }

    .badge {
      border-radius: 999px;
      padding: 7px 11px;
      font-weight: 800;
      font-size: 0.88rem;
    }

    .low { background: #dcfce7; color: var(--low); }
    .medium { background: #fef3c7; color: var(--medium); }
    .high { background: #fee2e2; color: var(--high); }

    .meter {
      width: 100%;
      height: 14px;
      border-radius: 999px;
      background: #ece6dc;
      overflow: hidden;
      border: 1px solid var(--line);
    }

    .meter > div {
      height: 100%;
      width: 0%;
      transition: width 0.4s ease;
      background: linear-gradient(90deg, #16a34a 0%, #eab308 55%, #dc2626 100%);
    }

    .muted { color: var(--muted); }

    ul {
      margin: 8px 0 0;
      padding-left: 18px;
      line-height: 1.6;
    }

    .factor-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin: 4px 6px 0 0;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 0.84rem;
    }

    .wide { grid-column: 1 / -1; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px 6px;
    }

    th { color: var(--muted); font-weight: 700; }

    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="container">
    <section class="hero">
      <h1>Mental Health Risk AI Dashboard</h1>
      <p class="subtitle">Portfolio-ready FastAPI application with AI prediction, explainability, recommendations, chart visualization, and history tracking.</p>
    </section>

    <section class="layout">
      <article class="card">
        <h2>Input Form</h2>
        <div class="grid">
          <div class="field"><label>Sleep Hours</label><input id="sleep_hours" type="number" min="0" max="24" step="0.1" value="6.5" /></div>
          <div class="field"><label>Stress Level (0-10)</label><input id="stress_level" type="number" min="0" max="10" step="1" value="8" /></div>
          <div class="field"><label>Work Hours</label><input id="work_hours" type="number" min="0" max="24" step="0.1" value="9" /></div>
          <div class="field"><label>Social Activity</label><input id="social_activity" type="number" min="0" step="0.1" value="2.5" /></div>
          <div class="field"><label>Physical Activity</label><input id="physical_activity" type="number" min="0" step="0.1" value="3.0" /></div>
          <div class="field"><label>Screen Time</label><input id="screen_time" type="number" min="0" max="24" step="0.1" value="7.0" /></div>
        </div>
        <div class="actions">
          <button class="primary" onclick="predictRisk()">Predict Risk</button>
          <button class="secondary" onclick="loadSample()">Load Sample</button>
          <button class="secondary" onclick="loadHistory()">Refresh History</button>
        </div>
      </article>

      <article class="card">
        <h2>Prediction Result</h2>
        <div class="result-header">
          <span id="riskBadge" class="badge medium">Awaiting prediction</span>
          <strong id="riskScoreText">Risk Score: --%</strong>
        </div>
        <div class="meter"><div id="riskMeterFill"></div></div>
        <p id="explanation" class="muted">Run a prediction to get model explanation.</p>
        <div id="factors"></div>

        <h3>Recommendations</h3>
        <ul id="recommendations">
          <li class="muted">Personalized recommendations will appear here.</li>
        </ul>
      </article>

      <article class="card">
        <h2>Input Profile Chart</h2>
        <canvas id="inputRadar"></canvas>
      </article>

      <article class="card">
        <h2>Feature Importance</h2>
        <canvas id="importanceBar"></canvas>
      </article>

      <article class="card wide">
        <h2>Prediction History</h2>
        <div style="display:grid;grid-template-columns:1.1fr 0.9fr;gap:16px;">
          <div>
            <table>
              <thead>
                <tr>
                  <th>Time</th><th>Score</th><th>Level</th><th>Stress</th><th>Sleep</th>
                </tr>
              </thead>
              <tbody id="historyRows">
                <tr><td colspan="5" class="muted">No history yet.</td></tr>
              </tbody>
            </table>
          </div>
          <div>
            <canvas id="trendChart"></canvas>
          </div>
        </div>
      </article>
    </section>
  </main>

  <script>
    let radarChart;
    let importanceChart;
    let trendChart;

    function getPayload() {
      return {
        sleep_hours: Number(document.getElementById('sleep_hours').value),
        stress_level: Number(document.getElementById('stress_level').value),
        work_hours: Number(document.getElementById('work_hours').value),
        social_activity: Number(document.getElementById('social_activity').value),
        physical_activity: Number(document.getElementById('physical_activity').value),
        screen_time: Number(document.getElementById('screen_time').value)
      };
    }

    function loadSample() {
      document.getElementById('sleep_hours').value = 6.5;
      document.getElementById('stress_level').value = 8;
      document.getElementById('work_hours').value = 9;
      document.getElementById('social_activity').value = 2.5;
      document.getElementById('physical_activity').value = 3;
      document.getElementById('screen_time').value = 7;
      drawInputRadar(getPayload());
    }

    function levelClass(level) {
      const v = String(level || '').toLowerCase();
      if (v.includes('low')) return 'low';
      if (v.includes('high')) return 'high';
      return 'medium';
    }

    function drawInputRadar(payload) {
      const labels = ['Sleep', 'Stress', 'Work', 'Social', 'Physical', 'Screen'];
      const values = [
        payload.sleep_hours,
        payload.stress_level,
        payload.work_hours,
        payload.social_activity,
        payload.physical_activity,
        payload.screen_time
      ];

      if (radarChart) radarChart.destroy();
      radarChart = new Chart(document.getElementById('inputRadar'), {
        type: 'radar',
        data: {
          labels,
          datasets: [{
            label: 'Input Profile',
            data: values,
            fill: true,
            backgroundColor: 'rgba(15, 118, 110, 0.2)',
            borderColor: 'rgba(15, 118, 110, 0.9)',
            borderWidth: 2,
            pointBackgroundColor: 'rgba(15, 118, 110, 1)'
          }]
        },
        options: {
          responsive: true,
          scales: { r: { beginAtZero: true } }
        }
      });
    }

    function drawImportanceBar(items) {
      const labels = items.map(x => x.feature);
      const data = items.map(x => x.importance);

      if (importanceChart) importanceChart.destroy();
      importanceChart = new Chart(document.getElementById('importanceBar'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Feature Importance',
            data,
            backgroundColor: 'rgba(202, 138, 4, 0.7)',
            borderColor: 'rgba(202, 138, 4, 1)',
            borderWidth: 1
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          plugins: { legend: { display: false } }
        }
      });
    }

    function drawTrend(historyItems) {
      const reversed = [...historyItems].reverse();
      const labels = reversed.map((x, i) => `#${i + 1}`);
      const scores = reversed.map(x => x.risk_score);

      if (trendChart) trendChart.destroy();
      trendChart = new Chart(document.getElementById('trendChart'), {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Risk Score Trend',
            data: scores,
            borderColor: 'rgba(185, 28, 28, 0.9)',
            backgroundColor: 'rgba(185, 28, 28, 0.2)',
            tension: 0.25,
            fill: true
          }]
        },
        options: {
          responsive: true,
          scales: { y: { min: 0, max: 100 } }
        }
      });
    }

    async function loadHistory() {
      try {
        const response = await fetch('/history?limit=10');
        const data = await response.json();
        const rows = document.getElementById('historyRows');

        if (!data.items || data.items.length === 0) {
          const msg = data.status === 'db_unavailable'
            ? 'History database unavailable in current environment.'
            : 'No history yet.';
          rows.innerHTML = `<tr><td colspan="5" class="muted">${msg}</td></tr>`;
          drawTrend([]);
          return;
        }

        rows.innerHTML = data.items.map(item => {
          const cls = levelClass(item.risk_level);
          return `<tr>
            <td>${item.created_at}</td>
            <td>${Number(item.risk_score).toFixed(1)}%</td>
            <td><span class="badge ${cls}">${item.risk_level}</span></td>
            <td>${item.inputs.stress_level}</td>
            <td>${item.inputs.sleep_hours}</td>
          </tr>`;
        }).join('');

        drawTrend(data.items);
      } catch (error) {
        console.error(error);
      }
    }

    async function predictRisk() {
      const payload = getPayload();
      drawInputRadar(payload);
      document.getElementById('explanation').textContent = 'Predicting...';

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Prediction failed');
        }

        const cls = levelClass(data.risk_level);
        document.getElementById('riskBadge').className = `badge ${cls}`;
        document.getElementById('riskBadge').textContent = `${data.risk_level} Risk`;
        document.getElementById('riskScoreText').textContent = `Risk Score: ${data.risk_score}%`;
        document.getElementById('riskMeterFill').style.width = `${data.risk_score}%`;
        const persistenceNote = data.persistence_status === 'save_failed'
          ? ' (Prediction shown, but history could not be saved.)'
          : '';
        document.getElementById('explanation').textContent = `${data.explanation}${persistenceNote}`;

        const factors = document.getElementById('factors');
        factors.innerHTML = (data.key_factors || []).map(f =>
          `<span class="factor-chip">${f.factor}: ${f.reason}</span>`
        ).join('');

        const rec = document.getElementById('recommendations');
        rec.innerHTML = (data.recommendations || []).map(x => `<li>${x}</li>`).join('');

        drawImportanceBar(data.feature_importance || []);
        loadHistory();
      } catch (error) {
        document.getElementById('explanation').textContent = `Error: ${error.message}`;
      }
    }

    window.addEventListener('load', () => {
      drawInputRadar(getPayload());
      drawImportanceBar([]);
      loadHistory();
    });
  </script>
</body>
</html>
""".strip()

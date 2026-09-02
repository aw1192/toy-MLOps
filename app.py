"""
Step 2: Serving layer.

Wraps model.pkl in an HTTP API. Every prediction gets logged to a local
SQLite file (predictions.db) -- this is the seed of your monitoring system.
A real system would ship these logs to a proper store (Postgres, a data
warehouse, a metrics platform), but SQLite demonstrates the exact same idea
at toy scale: every prediction leaves a trace you can look back at.

Run locally:
    uvicorn app:app --reload

Test it:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
        -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
"""
import json
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.pkl")
FEATURES_PATH = Path("feature_names.json")
DB_PATH = Path("predictions.db")

model = None
feature_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not per-request -- loading from disk
    on every call would tank latency. Modern FastAPI's replacement for the
    deprecated @app.on_event("startup") pattern."""
    global model, feature_names
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run `python train.py` first to produce it."
        )
    model = joblib.load(MODEL_PATH)
    feature_names = json.loads(FEATURES_PATH.read_text())
    init_db()
    yield
    # (no shutdown cleanup needed here)


app = FastAPI(title="Toy MLOps Model API", lifespan=lifespan)


def init_db():
    with get_db() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                features TEXT,
                prediction REAL,
                latency_ms REAL
            )
            """
        )


@contextmanager
def get_db():
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


class PredictRequest(BaseModel):
    # In production you'd usually accept named fields (age=.., amount=.., ...)
    # rather than a raw float list -- named fields validate against typos and
    # let Pydantic reject malformed requests before they ever hit the model.
    # Kept as a list here since this toy dataset has 30 numeric features and
    # naming each isn't the point of the exercise.
    features: list[float] = Field(..., min_length=30, max_length=30)


class PredictResponse(BaseModel):
    prediction: float
    latency_ms: float


@app.get("/health")
def health():
    """Deployment platforms (Render, Railway, k8s, ...) hit this repeatedly
    to check if the container is alive before routing traffic to it."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    prob = float(model.predict_proba([req.features])[0][1])
    latency_ms = (time.perf_counter() - start) * 1000

    # Log every prediction: input, output, latency, timestamp.
    # This is what makes drift detection and a monitoring dashboard possible later --
    # you can't detect that inputs are drifting if you never recorded what came in.
    with get_db() as con:
        con.execute(
            "INSERT INTO predictions (timestamp, features, prediction, latency_ms) VALUES (?, ?, ?, ?)",
            (time.time(), json.dumps(req.features), prob, latency_ms),
        )

    return PredictResponse(prediction=prob, latency_ms=latency_ms)

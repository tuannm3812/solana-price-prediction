from contextlib import asynccontextmanager
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.pipeline_components import SolanaFeatureEngineer
from solana_price_prediction.kraken import fetch_kraken_ohlc, kraken_since_timestamp

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/solana_next_day_high.joblib"))
KRAKEN_PAIR = os.getenv("KRAKEN_PAIR", "SOLUSD")
KRAKEN_INTERVAL = int(os.getenv("KRAKEN_INTERVAL", "1440"))
KRAKEN_LOOKBACK_DAYS = int(os.getenv("KRAKEN_LOOKBACK_DAYS", "180"))

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        ml_models["model"] = joblib.load(MODEL_PATH)
        print(f"Loaded model: {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}")
    yield
    ml_models.clear()


app = FastAPI(
    title="Solana High Price Predictor",
    description="Predicts Solana's next-day high price from recent OHLCV market history.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionOut(BaseModel):
    token: str
    prediction_date: str
    predicted_high: float


def fetch_history() -> pd.DataFrame:
    """Get recent OHLCV candles from Kraken."""
    try:
        since_timestamp = kraken_since_timestamp(KRAKEN_LOOKBACK_DAYS)
        return fetch_kraken_ohlc(
            pair=KRAKEN_PAIR,
            interval=KRAKEN_INTERVAL,
            since_timestamp=since_timestamp,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Data provider error: {exc}") from exc


@app.get("/health")
def health() -> dict[str, bool]:
    return {"model_loaded": "model" in ml_models}


@app.get("/predict/solana", response_model=PredictionOut)
def predict() -> dict[str, str | float]:
    if "model" not in ml_models:
        raise HTTPException(500, "Model not active")

    raw_df = fetch_history()
    try:
        features_df = SolanaFeatureEngineer().transform(raw_df)
    except Exception as exc:
        raise HTTPException(500, f"Pipeline error: {exc}") from exc

    prediction = ml_models["model"].predict(features_df.iloc[[-1]])[0]
    tomorrow = raw_df.iloc[-1]["timestamp"] + pd.Timedelta(days=1)

    return {
        "token": "SOL",
        "prediction_date": tomorrow.strftime("%Y-%m-%d"),
        "predicted_high": float(prediction),
    }

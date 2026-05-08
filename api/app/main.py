from contextlib import asynccontextmanager
import os
from pathlib import Path

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.pipeline_components import SolanaFeatureEngineer

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/xgboost_solana_v1.joblib"))
COINGECKO_API = "https://api.coingecko.com/api/v3/coins/solana/ohlc"
COINGECKO_CHART_API = "https://api.coingecko.com/api/v3/coins/solana/market_chart"

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
    """Get 90 days of OHLCV data from CoinGecko."""
    try:
        params = {"vs_currency": "usd", "days": "90"}
        resp = requests.get(COINGECKO_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

        chart_resp = requests.get(COINGECKO_CHART_API, params=params, timeout=10)
        chart_resp.raise_for_status()
        chart_data = chart_resp.json()
        volumes = pd.DataFrame(chart_data["total_volumes"], columns=["timestamp", "volume"])
        return df.merge(volumes, on="timestamp", how="left")
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
    tomorrow = raw_df.iloc[-1]["date"] + pd.Timedelta(days=1)

    return {
        "token": "SOL",
        "prediction_date": tomorrow.strftime("%Y-%m-%d"),
        "predicted_high": float(prediction),
    }

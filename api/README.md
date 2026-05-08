# Solana Prediction API

FastAPI service that loads the trained XGBoost model and predicts Solana's next-day high price from recent CoinGecko OHLCV data.

## Run Locally

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `GET /predict/solana`
- `GET /docs`

Set a custom model path with:

```bash
$env:MODEL_PATH="models/xgboost_solana_v1.joblib"
```

The API imports feature engineering from the root package so training and serving use the same feature definitions.

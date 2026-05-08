# Solana Prediction API

FastAPI service that loads the trained model and predicts Solana's next-day high price from recent Kraken OHLCV data.

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
$env:MODEL_PATH="models/solana_next_day_high.joblib"
```

Optional Kraken settings:

```bash
$env:KRAKEN_PAIR="SOLUSD"
$env:KRAKEN_INTERVAL="1440"
$env:KRAKEN_LOOKBACK_DAYS="180"
```

The API imports feature engineering from the root package so training and serving use the same feature definitions.

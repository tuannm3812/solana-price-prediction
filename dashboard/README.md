# Solana Streamlit Dashboard

Streamlit dashboard for live Solana market data and next-day high predictions.

## Local Run

From the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run dashboard/app/main.py
```

The dashboard works without a separate API. It fetches Kraken OHLCV candles and loads the bundled model from `models/solana_next_day_high.joblib`.

## Optional API

If you deploy the FastAPI service separately, add this Streamlit secret:

```toml
PREDICTION_API_URL = "https://your-api.example.com"
```

When configured, the dashboard calls the API first and falls back to the bundled model if the API is unavailable.

## Streamlit Community Cloud

- Main file path: `dashboard/app/main.py`
- Python version: `3.11`
- Secrets: none required

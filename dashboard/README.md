# Crypto Investment Dashboard

Streamlit dashboard for viewing Solana market data and requesting next-day high predictions from the FastAPI service.

## Run Locally

Start the API first, then run:

```bash
python -m pip install -r requirements.txt
streamlit run app/main.py
```

The app uses `http://127.0.0.1:8000` by default. Override the API URL with:

```bash
$env:PREDICTION_API_URL="https://your-api.example.com"
```

Market data comes from Kraken. Predictions come from `GET /predict/solana`.

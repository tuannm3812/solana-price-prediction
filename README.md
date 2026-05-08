# Solana Price Prediction

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-46A5D1)](https://docs.astral.sh/ruff/)

![Solana hero image](docs/assets/solana-hero.jpg)

<sub>Image: [Solana cryptocurrency two.jpg](https://commons.wikimedia.org/wiki/File:Solana_cryptocurrency_two.jpg) by Clearus, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).</sub>

Production-ready Python project for predicting Solana's next-day high price from OHLCV market data. The repository contains a reusable modeling package, a FastAPI prediction service, a Streamlit dashboard, and the original notebook retained as experiment history.

## What This Project Does

- Builds a cleaned Solana modeling dataset from a downloadable URL or local CSV exports.
- Generates technical indicators: SMA, RSI, volatility, and lagged price/volume features.
- Trains an XGBoost regressor with chronological train/validation/test splitting.
- Serves live predictions through FastAPI using CoinGecko market data.
- Displays Kraken market data and API predictions in Streamlit.

## Repository Layout

```text
solana-price-prediction/
|-- solana_price_prediction/     # Reusable Python package
|   |-- dataset.py               # Raw CSV loading and cleaning
|   |-- features.py              # Feature engineering shared by training and API
|   `-- modeling/
|       |-- train.py             # XGBoost training CLI
|       `-- predict.py           # Batch inference CLI
|-- api/                         # FastAPI service
|-- dashboard/                   # Streamlit dashboard
|-- notebooks/                   # Original exploratory notebook
|-- models/                      # Trained model artifacts
|-- tests/                       # Unit tests
|-- pyproject.toml               # Package metadata and dependencies
`-- Makefile                     # Common developer commands
```

## Quick Start

Create an environment with Python 3.11, then install the project:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Run the tests:

```bash
pytest
```

## Training Pipeline

Use a downloadable CSV, JSON, or parquet URL:

```bash
python -m solana_price_prediction.dataset --input-url "https://example.com/solana.csv"
python -m solana_price_prediction.features
python -m solana_price_prediction.modeling.train
```

Or place raw Solana CSV files under `data/raw/Solana/`, then run:

```bash
python -m solana_price_prediction.dataset
python -m solana_price_prediction.features
python -m solana_price_prediction.modeling.train
```

Training uses chronological splits by default:

- 70% train
- 15% validation for early stopping
- 15% test for final reporting

Override the split ratios when needed:

```bash
python -m solana_price_prediction.modeling.train --validation-size 0.2 --test-size 0.2
```

Default outputs:

- Clean dataset: `data/processed/solana_model_data.parquet`
- Feature table: `data/processed/solana_features.parquet`
- Model artifact: `models/xgboost_solana_v1.joblib`
- Test predictions: `data/processed/solana_predictions.parquet`

The `data/` directory is intentionally ignored by Git.

## Run The API

```bash
cd api
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Useful endpoints:

- `GET /health`
- `GET /predict/solana`
- `GET /docs`

## Run The Dashboard

In a separate terminal:

```bash
cd dashboard
python -m pip install -r requirements.txt
streamlit run app/main.py
```

The dashboard expects the API at `http://127.0.0.1:8000` by default. Override it with:

```bash
$env:PREDICTION_API_URL="https://your-api.example.com"
```

## From Notebook To Project

The notebook logic has been moved into importable modules:

- Data loading and cleaning: `solana_price_prediction.dataset`
- Target and feature creation: `solana_price_prediction.features`
- Model training and metrics: `solana_price_prediction.modeling.train`
- Inference: `solana_price_prediction.modeling.predict`

The notebook remains in `notebooks/` as an experiment record, while production code lives in the package and is covered by tests.

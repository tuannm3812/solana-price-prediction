import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from utils import fetch_kraken_data  # noqa: E402

from solana_price_prediction.modeling.predict import predict_next_high  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "solana_next_day_high.joblib"


def _prediction_api_url() -> str | None:
    """Return an optional deployed API URL from Streamlit secrets."""
    try:
        return st.secrets.get("PREDICTION_API_URL")
    except FileNotFoundError:
        return os.getenv("PREDICTION_API_URL")


@st.cache_resource
def _load_local_model():
    return joblib.load(MODEL_PATH)


def render_solana_tab() -> None:
    st.markdown("## Solana (SOL) Intelligence")
    st.markdown("Live market data with an Anchored Residual next-day high forecast.")
    st.divider()

    with st.spinner("Fetching live market data..."):
        df = fetch_kraken_data("SOLUSD", interval=1440)

    if not df.empty:
        _render_market_summary(df)
        _render_price_chart(df)
    else:
        st.error("Unable to load market data. Please try again later.")

    st.divider()
    _render_prediction_panel(df)


def _render_market_summary(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    price = latest["close"]
    pct_change = ((price - previous["close"]) / previous["close"]) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${price:,.2f}", f"{pct_change:+.2f}%")
    m2.metric("24h High", f"${latest['high']:,.2f}")
    m3.metric("24h Low", f"${latest['low']:,.2f}")
    m4.metric("Volume", f"{latest['volume']:,.0f}")


def _render_price_chart(df: pd.DataFrame) -> None:
    st.subheader("Price History")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"].rolling(30).mean(),
            mode="lines",
            name="SMA 30",
            line={"color": "#f59e0b", "width": 2},
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Volume",
            marker_color="rgba(148, 163, 184, 0.35)",
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"orientation": "h", "y": 1, "x": 0},
        yaxis={"title": "Price (USD)", "side": "left"},
        yaxis2={
            "title": "Volume",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "range": [0, df["volume"].max() * 4],
        },
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_prediction_panel(df: pd.DataFrame) -> None:
    st.subheader("Model Prediction")
    c1, c2 = st.columns([1, 2])

    with c1:
        st.info(
            "Model: anchored residual regressor\n\n"
            "Target: next-day high\n\n"
            "Signals: RSI, SMA, volatility, lags"
        )
        if st.button("Predict Tomorrow's High", type="primary", use_container_width=True):
            with st.spinner("Generating prediction..."):
                try:
                    st.session_state["pred_result"] = _predict_from_api_or_local(df)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    with c2:
        if "pred_result" not in st.session_state:
            st.caption("Run a prediction to show the latest model output.")
            return

        result = st.session_state["pred_result"]
        predicted_high = result["predicted_high"]
        st.success(f"Prediction for {result['prediction_date']}")
        st.metric("Predicted High Price", f"${predicted_high:,.2f}")

        if not df.empty:
            current_high = df.iloc[-1]["high"]
            diff = predicted_high - current_high
            direction = "higher" if diff > 0 else "lower"
            st.caption(f"Prediction is ${abs(diff):.2f} {direction} than the latest daily high.")


def _predict_from_api_or_local(df: pd.DataFrame) -> dict[str, str | float]:
    api_url = _prediction_api_url()
    if api_url:
        try:
            response = requests.get(f"{api_url.rstrip('/')}/predict/solana", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            st.warning(f"Prediction API unavailable, using bundled model instead. Details: {exc}")

    if df.empty:
        raise ValueError("Cannot run local prediction without market data.")

    model = _load_local_model()
    raw_history = df.rename(columns={"date": "timestamp"}).copy()
    raw_history["marketcap"] = 0.0
    prediction = predict_next_high(model, raw_history)
    next_date = pd.to_datetime(raw_history.iloc[-1]["timestamp"]) + pd.Timedelta(days=1)
    return {
        "token": "SOL",
        "prediction_date": next_date.strftime("%Y-%m-%d"),
        "predicted_high": prediction,
    }

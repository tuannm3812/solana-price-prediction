import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (APP_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import fetch_kraken_data  # noqa: E402

from solana_price_prediction.modeling.predict import predict_next_high  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "solana_next_day_high.joblib"
LOOKBACK_OPTIONS = {
    "90D": 90,
    "180D": 180,
    "1Y": 365,
    "All": None,
}


def _prediction_api_url() -> str | None:
    """Return an optional deployed API URL from Streamlit secrets."""
    env_url = os.getenv("PREDICTION_API_URL")
    if env_url:
        return env_url

    try:
        return st.secrets.get("PREDICTION_API_URL")
    except (FileNotFoundError, KeyError):
        return None


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
    m4.metric("Volume", f"{_format_compact_number(latest['volume'])} SOL")


def _render_price_chart(df: pd.DataFrame) -> None:
    st.subheader("Price History")

    control_1, control_2, control_3 = st.columns([1, 1, 2])
    lookback_label = control_1.selectbox(
        "Range",
        options=list(LOOKBACK_OPTIONS),
        index=2,
        label_visibility="collapsed",
        key="sol_chart_range",
    )
    show_sma = control_2.checkbox("SMA 30", value=True, key="sol_show_sma")
    show_volume = control_3.checkbox("Volume", value=True, key="sol_show_volume")

    chart_df = _filter_lookback(df, LOOKBACK_OPTIONS[lookback_label])
    row_heights = [0.76, 0.24] if show_volume else [1.0]
    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )
    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    if show_sma:
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=chart_df["close"].rolling(30).mean(),
                mode="lines",
                name="SMA 30",
                line={"color": "#f59e0b", "width": 2},
            ),
            row=1,
            col=1,
        )
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=chart_df["date"],
                y=chart_df["volume"],
                name="Volume",
                marker_color="rgba(20, 241, 149, 0.28)",
                hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f} SOL<extra>Volume</extra>",
            ),
            row=2,
            col=1,
        )

        average_volume = chart_df["volume"].mean()
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=[average_volume] * len(chart_df),
                mode="lines",
                name="Avg Volume",
                line={"color": "rgba(148, 163, 184, 0.85)", "width": 1, "dash": "dot"},
                hovertemplate="%{y:,.0f} SOL<extra>Avg Volume</extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=560,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"orientation": "h", "y": 1, "x": 0},
        hovermode="x unified",
        bargap=0,
        yaxis={"title": "Price (USD)", "side": "left"},
    )
    if show_volume:
        fig.update_yaxes(title_text="Volume (SOL)", tickformat=".2s", row=2, col=1)
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

        if not df.empty:
            current_high = df.iloc[-1]["high"]
            diff = predicted_high - current_high
            pct_diff = (diff / current_high) * 100
            _render_prediction_delta(predicted_high, diff, pct_diff)


def _render_prediction_delta(predicted_high: float, diff: float, pct_diff: float) -> None:
    color = "#22c55e" if diff >= 0 else "#ef4444"
    bg_color = "rgba(34, 197, 94, 0.14)" if diff >= 0 else "rgba(239, 68, 68, 0.14)"
    arrow = "&#8593;" if diff >= 0 else "&#8595;"
    label = "above" if diff >= 0 else "below"
    st.markdown(
        f"""
        <div style="margin-top: 0.25rem;">
            <div style="font-size: 0.875rem; font-weight: 700;">Predicted High Price</div>
            <div style="display: flex; align-items: baseline; gap: 0.75rem; flex-wrap: wrap;">
                <span style="font-size: 2.25rem; line-height: 1.2;">${predicted_high:,.2f}</span>
                <span style="
                    color: {color};
                    background: {bg_color};
                    border: 1px solid {color};
                    border-radius: 999px;
                    padding: 0.2rem 0.55rem;
                    font-size: 0.95rem;
                    font-weight: 700;
                    white-space: nowrap;
                ">
                    {arrow} ${abs(diff):,.2f} ({pct_diff:+.2f}%)
                </span>
            </div>
            <div style="color: #cbd5e1; margin-top: 0.7rem;">
                Prediction is {label} the latest daily high.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _filter_lookback(df: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if days is None or df.empty:
        return df

    cutoff = pd.to_datetime(df["date"]).max() - pd.Timedelta(days=days)
    return df[pd.to_datetime(df["date"]) >= cutoff]


def _format_compact_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


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

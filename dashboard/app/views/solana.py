import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (APP_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import fetch_kraken_data, fetch_solana_news  # noqa: E402

from solana_price_prediction.modeling.predict import predict_next_high  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "solana_next_day_high.joblib"
LOOKBACK_OPTIONS = {
    "90D": 90,
    "180D": 180,
    "1Y": 365,
    "All": None,
}


def _prediction_api_url() -> str | None:
    """Return an optional deployed API URL from the environment."""
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
        _render_fetch_status(df)
        _render_market_summary(df)
        _render_price_chart(df)
        _render_technical_summary(df)
        _render_news()
    else:
        st.error("Unable to load market data. Please try again later.")

    st.divider()
    _render_prediction_panel(df)


def _render_market_summary(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    price = latest["close"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Current Price",
        f"${price:,.2f}",
        _format_price_delta(price, previous["close"]),
    )
    m2.metric(
        "24h High",
        f"${latest['high']:,.2f}",
        _format_price_delta(latest["high"], previous["high"]),
    )
    m3.metric(
        "24h Low",
        f"${latest['low']:,.2f}",
        _format_price_delta(latest["low"], previous["low"]),
    )
    m4.metric(
        "Volume",
        f"{_format_compact_number(latest['volume'])} SOL",
        _format_volume_delta(latest["volume"], previous["volume"]),
    )


def _render_fetch_status(df: pd.DataFrame) -> None:
    fetched_at = df.attrs.get("fetched_at_utc")
    latest_candle = pd.to_datetime(df.iloc[-1]["date"])
    if fetched_at:
        fetched_text = pd.to_datetime(fetched_at).strftime("%Y-%m-%d %H:%M UTC")
        st.caption(
            f"Fetched from Kraken at {fetched_text}. "
            f"Latest daily candle: {latest_candle:%Y-%m-%d}."
        )
    else:
        st.caption(f"Latest daily candle: {latest_candle:%Y-%m-%d}.")


def _render_price_chart(df: pd.DataFrame) -> None:
    st.subheader("Price History")

    control_1, control_2, control_3, control_4 = st.columns([1, 1, 1, 2])
    lookback_label = control_1.selectbox(
        "Range",
        options=list(LOOKBACK_OPTIONS),
        index=2,
        label_visibility="collapsed",
        key="sol_chart_range",
    )
    show_sma = control_2.checkbox("SMA 30", value=True, key="sol_show_sma")
    show_volume = control_3.checkbox("Volume", value=True, key="sol_show_volume")
    show_levels = control_4.checkbox("Slope and peaks", value=True, key="sol_show_levels")

    chart_df = _filter_lookback(df, LOOKBACK_OPTIONS[lookback_label])
    fig = go.Figure()
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=chart_df["date"],
                y=chart_df["volume"],
                name="Volume",
                marker_color="rgba(20, 241, 149, 0.22)",
                hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f} SOL<extra>Volume</extra>",
                yaxis="y2",
            )
        )
    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
        )
    )
    if show_sma:
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=chart_df["close"].rolling(30).mean(),
                mode="lines",
                name="SMA 30",
                line={"color": "#f59e0b", "width": 2},
            )
        )

    if show_levels:
        _add_slope_and_peak_traces(fig, chart_df)

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
        volume_max = chart_df["volume"].max()
        fig.update_layout(
            yaxis2={
                "title": "Volume (SOL)",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "tickformat": ".2s",
                "range": [0, volume_max * 3 if volume_max else 1],
            }
        )
    st.plotly_chart(fig, use_container_width=True)


def _add_slope_and_peak_traces(fig: go.Figure, chart_df: pd.DataFrame) -> None:
    if len(chart_df) < 7:
        return

    recent = chart_df.tail(min(14, len(chart_df)))
    fig.add_trace(
        go.Scatter(
            x=[recent.iloc[0]["date"], recent.iloc[-1]["date"]],
            y=[recent.iloc[0]["close"], recent.iloc[-1]["close"]],
            mode="lines+markers",
            name="Short Slope",
            line={"color": "#38bdf8", "width": 2, "dash": "dot"},
            marker={"size": 7},
        )
    )

    recent_window = chart_df.tail(min(30, len(chart_df)))
    peak = recent_window.loc[recent_window["high"].idxmax()]
    trough = recent_window.loc[recent_window["low"].idxmin()]
    fig.add_trace(
        go.Scatter(
            x=[peak["date"]],
            y=[peak["high"]],
            mode="markers+text",
            name="30D Peak",
            marker={"color": "#facc15", "size": 10, "symbol": "triangle-up"},
            text=["30D peak"],
            textposition="top center",
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra>30D Peak</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trough["date"]],
            y=[trough["low"]],
            mode="markers+text",
            name="30D Low",
            marker={"color": "#fb7185", "size": 10, "symbol": "triangle-down"},
            text=["30D low"],
            textposition="bottom center",
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra>30D Low</extra>",
        )
    )


def _render_technical_summary(df: pd.DataFrame) -> None:
    st.subheader("Key Signals")
    latest = _calculate_indicators(df).iloc[-1]
    peak_30 = df.tail(30)["high"].max()
    drawdown = ((latest["close"] - peak_30) / peak_30) * 100 if peak_30 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RSI 14", f"{latest['rsi_14']:.1f}", _rsi_label(latest["rsi_14"]))
    c2.metric(
        "SMA 7 vs 30",
        f"{latest['sma_7'] - latest['sma_30']:+.2f}",
        "short trend gap",
    )
    c3.metric("7D Slope", f"{latest['slope_7_pct']:+.2f}%", "close-to-close")
    c4.metric("30D Volatility", f"{latest['volatility_30']:.1f}%", "annualized")
    c5.metric("From 30D Peak", f"{drawdown:+.2f}%", f"peak ${peak_30:,.2f}")


def _render_news() -> None:
    st.subheader("Solana News")
    news = fetch_solana_news()
    if not news:
        st.caption("News feed unavailable right now.")
        return

    for item in news[:5]:
        st.markdown(f"- [{item['title']}]({item['url']}) · {item['source']}")


def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["sma_7"] = result["close"].rolling(7).mean()
    result["sma_30"] = result["close"].rolling(30).mean()
    result["slope_7_pct"] = result["close"].pct_change(7) * 100
    result["volatility_30"] = result["close"].pct_change().rolling(30).std() * (365**0.5) * 100

    delta = result["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask(avg_loss == 0, 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    result["rsi_14"] = rsi
    return result.ffill().fillna(0).infer_objects(copy=False)


def _rsi_label(value: float) -> str:
    if value >= 70:
        return "overbought"
    if value <= 30:
        return "oversold"
    return "neutral"


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


def _format_price_delta(current: float, previous: float) -> str:
    if not previous:
        return "$0.00 (0.00%)"
    diff = current - previous
    pct = (diff / previous) * 100
    sign = "+" if diff >= 0 else "-"
    return f"{sign}${abs(diff):,.2f} ({pct:+.2f}%)"


def _format_volume_delta(current: float, previous: float) -> str:
    if not previous:
        return "0 SOL (0.00%)"
    diff = current - previous
    pct = (diff / previous) * 100
    return f"{_format_signed_compact_number(diff)} SOL ({pct:+.2f}%)"


def _format_signed_compact_number(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{_format_compact_number(abs(value))}"


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

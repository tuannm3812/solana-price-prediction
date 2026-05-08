import pandas as pd
import streamlit as st

from solana_price_prediction.kraken import fetch_kraken_ohlc


@st.cache_data(ttl=60)
def fetch_kraken_data(pair: str = "SOLUSD", interval: int = 1440) -> pd.DataFrame:
    """Fetch OHLCV data from Kraken."""
    try:
        df = fetch_kraken_ohlc(pair=pair, interval=interval, timeout=5)
        return df.rename(columns={"timestamp": "date"})[
            ["date", "open", "high", "low", "close", "volume"]
        ]
    except Exception as exc:
        st.error(f"Failed to fetch data: {exc}")
        return pd.DataFrame()

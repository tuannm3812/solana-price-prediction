# app/utils.py
import pandas as pd
import requests
import streamlit as st

# Kraken Public Endpoint
KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC"

@st.cache_data(ttl=60)
def fetch_kraken_data(pair: str = "SOLUSD", interval: int = 1440) -> pd.DataFrame:
    """Fetch OHLCV data from Kraken."""
    params = {"pair": pair, "interval": interval}
    
    try:
        response = requests.get(KRAKEN_OHLC, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("error"):
            st.error(f"Kraken API Error: {data['error']}")
            return pd.DataFrame()
        
        # Result key is dynamic (e.g., 'XXBTZUSD'), so we get the first key that isn't 'last'
        res = data["result"]
        pair_key = [k for k in res.keys() if k != "last"][0]
        ohlc = res[pair_key]
        
        # Kraken returns: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(ohlc, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
        
        # Convert types
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # Convert Date
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        
        return df[["date", "open", "high", "low", "close", "volume"]]
        
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return pd.DataFrame()

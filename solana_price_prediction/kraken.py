from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
KRAKEN_OHLC_COLUMNS = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]


def kraken_since_timestamp(lookback_days: int) -> int:
    """Return a UTC Unix timestamp for a Kraken OHLC lookback window."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be greater than zero.")
    return int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())


def parse_kraken_ohlc(payload: dict) -> pd.DataFrame:
    """Convert a Kraken OHLC API payload into a typed DataFrame."""
    if payload.get("error"):
        raise ValueError(f"Kraken API returned errors: {payload['error']}")
    if "result" not in payload:
        raise ValueError("Kraken API response is missing the result field.")

    pair_keys = [key for key in payload["result"] if key != "last"]
    if not pair_keys:
        raise ValueError("Kraken API response does not contain OHLC rows.")

    df = pd.DataFrame(payload["result"][pair_keys[0]], columns=KRAKEN_OHLC_COLUMNS)
    for column in ["open", "high", "low", "close", "vwap", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["marketcap"] = 0.0
    return df


def fetch_kraken_ohlc(
    pair: str = "SOLUSD",
    interval: int = 1440,
    since_timestamp: int | None = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """Fetch OHLCV candles from Kraken's public OHLC endpoint."""
    params: dict[str, int | str] = {"pair": pair, "interval": interval}
    if since_timestamp is not None:
        params["since"] = since_timestamp

    response = requests.get(KRAKEN_OHLC_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return parse_kraken_ohlc(response.json())

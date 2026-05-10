import sys
from pathlib import Path
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solana_price_prediction.kraken import fetch_kraken_ohlc


@st.cache_data(ttl=60)
def fetch_kraken_data(pair: str = "SOLUSD", interval: int = 1440) -> pd.DataFrame:
    """Fetch OHLCV data from Kraken."""
    try:
        df = fetch_kraken_ohlc(pair=pair, interval=interval, timeout=5)
        result = df.rename(columns={"timestamp": "date"})[
            ["date", "open", "high", "low", "close", "volume"]
        ]
        result.attrs["fetched_at_utc"] = datetime.now(timezone.utc)
        return result
    except Exception as exc:
        st.error(f"Failed to fetch data: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=900)
def fetch_solana_news(limit: int = 5) -> list[dict[str, str]]:
    """Fetch recent Solana headlines from a no-key RSS source."""
    url = (
        "https://news.google.com/rss/search?"
        "q=Solana%20SOL%20cryptocurrency&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []

    items = []
    for item in root.findall("./channel/item")[:limit]:
        source = item.find("source")
        items.append(
            {
                "title": item.findtext("title", default="Untitled"),
                "url": item.findtext("link", default=""),
                "source": source.text if source is not None and source.text else "News",
                "published": item.findtext("pubDate", default=""),
            }
        )
    return items

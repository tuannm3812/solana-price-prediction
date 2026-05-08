import csv
import io
from pathlib import Path

import pandas as pd
import requests
import typer

from solana_price_prediction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer(help="Build cleaned Solana modeling datasets from local files or URLs.")


def sniff_delimiter(sample: str) -> str:
    """Infer the delimiter for a CSV sample."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_csv_text_robust(raw_bytes: bytes) -> pd.DataFrame:
    """Read CSV bytes with common delimiter and encoding fallbacks."""
    if len(raw_bytes) < 10:
        return pd.DataFrame()

    for encoding in ["utf-8", "cp1252", "latin1"]:
        try:
            text = raw_bytes.decode(encoding)
            return pd.read_csv(
                io.StringIO(text),
                sep=sniff_delimiter(text[:1024]),
                engine="python",
                on_bad_lines="skip",
            )
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    return pd.DataFrame()


def read_csv_robust(file_path: Path) -> pd.DataFrame:
    """Read a local CSV with common delimiter and encoding fallbacks."""
    return read_csv_text_robust(file_path.read_bytes())


def load_solana_data_from_url(url: str) -> pd.DataFrame:
    """Load Solana market data from a downloadable CSV, JSON, or parquet URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(response.content))
    if suffix == ".json":
        return pd.read_json(io.BytesIO(response.content))

    csv_df = read_csv_text_robust(response.content)
    if csv_df.empty:
        raise ValueError(f"No valid tabular data could be loaded from {url}")
    return csv_df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names used across training, API, and dashboard code."""
    normalized = df.copy()
    normalized.columns = normalized.columns.str.strip().str.lower().str.replace(" ", "_")
    return normalized.rename(
        columns={
            "market_cap": "marketcap",
            "market_capitalization": "marketcap",
            "volume_24h": "volume",
            "circulating_supply": "supply",
        }
    )


def load_solana_data(data_dir: Path | None = None, url: str | None = None) -> pd.DataFrame:
    """Load Solana data from a URL or combine all CSV files from a directory."""
    if url:
        return normalize_column_names(load_solana_data_from_url(url))

    if data_dir is None:
        raise ValueError("Provide either data_dir or url.")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for file_path in csv_files:
        raw_df = read_csv_robust(file_path)
        if not raw_df.empty:
            frames.append(normalize_column_names(raw_df))

    if not frames:
        raise ValueError(f"No valid Solana CSV data could be loaded from {data_dir}")

    return pd.concat(frames, ignore_index=True)


def clean_solana_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Solana OHLCV rows into a time-sorted modeling dataset."""
    df_clean = normalize_column_names(df)
    date_col = "timestamp" if "timestamp" in df_clean.columns else "timeopen"
    if date_col not in df_clean.columns:
        raise ValueError("Expected a timestamp or timeopen column in the raw dataset.")

    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    for column in ["open", "high", "low", "close", "volume", "marketcap"]:
        if column in df_clean.columns:
            if df_clean[column].dtype == "object":
                df_clean[column] = df_clean[column].astype(str).str.replace(r"[$,]", "", regex=True)
            df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")

    return df_clean.ffill()


@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR / "Solana",
    input_url: str | None = typer.Option(
        None,
        help="Optional downloadable CSV, JSON, or parquet URL. Takes precedence over input_dir.",
    ),
    output_path: Path = PROCESSED_DATA_DIR / "solana_model_data.parquet",
) -> None:
    """Load Solana market data and write a cleaned parquet dataset."""
    df = clean_solana_data(load_solana_data(input_dir, url=input_url))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    typer.echo(f"Wrote {len(df):,} cleaned rows to {output_path}")


if __name__ == "__main__":
    app()

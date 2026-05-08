from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import typer

from solana_price_prediction.config import PROCESSED_DATA_DIR

TARGET_COLUMN = "target_next_day_high"
DATE_COLUMNS = ["timestamp", "timeopen", "timeclose", "timehigh", "timelow", "date"]
NON_FEATURE_COLUMNS = [TARGET_COLUMN, *DATE_COLUMNS, "name", "symbol"]
EXPECTED_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "marketcap",
    "sma_7",
    "sma_30",
    "volatility_7",
    "rsi_14",
    "high_lag_1",
    "vol_lag_1",
    "high_lag_2",
    "vol_lag_2",
    "high_lag_3",
    "vol_lag_3",
    "high_lag_7",
    "vol_lag_7",
]

app = typer.Typer(help="Generate model features for the Solana price model.")


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index used by the training and API pipelines."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def create_target_variable(df: pd.DataFrame, target_col: str = "high") -> pd.DataFrame:
    """Create the next-day high target and remove the final row with no known target."""
    df_target = df.copy()
    df_target[TARGET_COLUMN] = df_target[target_col].shift(-1)
    return df_target.dropna(subset=[TARGET_COLUMN])


def add_features(df: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
    """Generate technical indicators and lag features from OHLCV market data."""
    df_feat = df.copy()
    df_feat.columns = df_feat.columns.str.strip().str.lower()

    if "marketcap" not in df_feat.columns:
        df_feat["marketcap"] = 0.0

    df_feat["sma_7"] = df_feat["close"].rolling(window=7).mean()
    df_feat["sma_30"] = df_feat["close"].rolling(window=30).mean()
    df_feat["volatility_7"] = df_feat["close"].rolling(window=7).std()
    df_feat["rsi_14"] = calculate_rsi(df_feat["close"], window=14)

    for lag in [1, 2, 3, 7]:
        df_feat[f"high_lag_{lag}"] = df_feat["high"].shift(lag)
        df_feat[f"vol_lag_{lag}"] = df_feat["volume"].shift(lag)

    if drop_missing:
        return df_feat.dropna().reset_index(drop=True)

    return df_feat.ffill().bfill().fillna(0.0)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns while excluding dates, labels, and identifiers."""
    return [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


class SolanaFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible feature transformer for online Solana inference."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SolanaFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = add_features(X, drop_missing=False)
        missing = [column for column in EXPECTED_FEATURES if column not in features.columns]
        for column in missing:
            features[column] = 0.0
        return features[EXPECTED_FEATURES]


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "solana_model_data.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "solana_features.parquet",
) -> None:
    """Generate and persist feature-ready training data."""
    df = pd.read_parquet(input_path)
    features = add_features(create_target_variable(df))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    typer.echo(f"Wrote {len(features):,} feature rows to {output_path}")


if __name__ == "__main__":
    app()

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import typer
from xgboost import XGBRegressor

from solana_price_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR
from solana_price_prediction.features import TARGET_COLUMN, select_feature_columns

app = typer.Typer(help="Train the production Solana next-day high model.")


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    baseline_mae: float


def split_time_series(
    df: pd.DataFrame,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically into train, validation, and test datasets."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1.")

    train_end = int(len(df) * (1 - validation_size - test_size))
    validation_end = int(len(df) * (1 - test_size))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:validation_end].copy(),
        df.iloc[validation_end:].copy(),
    )


def evaluate_baseline(test_df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> float:
    """Evaluate the persistence baseline: tomorrow's high equals today's high."""
    return mean_absolute_error(test_df[target_col], test_df["high"])


def train_xgboost(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> tuple[XGBRegressor, pd.DataFrame, RegressionMetrics]:
    """Train and evaluate the XGBoost regressor."""
    features = select_feature_columns(train_df)
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_validation = validation_df[features]
    y_validation = validation_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)

    predictions = model.predict(X_test)
    results = test_df.copy()
    results["predicted_high"] = predictions
    results["error"] = results[target_col] - results["predicted_high"]

    metrics = RegressionMetrics(
        mae=mean_absolute_error(y_test, predictions),
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        r2=r2_score(y_test, predictions),
        baseline_mae=evaluate_baseline(test_df, target_col),
    )
    return model, results, metrics


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "solana_features.parquet",
    model_path: Path = MODELS_DIR / "xgboost_solana_v1.joblib",
    results_path: Path = PROCESSED_DATA_DIR / "solana_predictions.parquet",
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> None:
    """Train the model from feature-ready parquet data."""
    df = pd.read_parquet(features_path)
    train_df, validation_df, test_df = split_time_series(
        df,
        validation_size=validation_size,
        test_size=test_size,
    )
    model, results, metrics = train_xgboost(train_df, validation_df, test_df)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    results.to_parquet(results_path, index=False)

    typer.echo(f"Model saved to {model_path}")
    typer.echo(
        "Split sizes: "
        f"train={len(train_df):,}, validation={len(validation_df):,}, test={len(test_df):,}"
    )
    typer.echo(
        "Metrics: "
        f"MAE=${metrics.mae:.4f}, RMSE=${metrics.rmse:.4f}, "
        f"R2={metrics.r2:.4f}, baseline MAE=${metrics.baseline_mae:.4f}"
    )


if __name__ == "__main__":
    app()

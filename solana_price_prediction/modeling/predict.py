from pathlib import Path

import joblib
import pandas as pd
import typer

from solana_price_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR
from solana_price_prediction.features import EXPECTED_FEATURES, SolanaFeatureEngineer

app = typer.Typer(help="Run batch inference with the trained Solana model.")


def predict_next_high(model, raw_history: pd.DataFrame) -> float:
    """Predict the next-day high from recent OHLCV history."""
    features = SolanaFeatureEngineer().transform(raw_history)
    return float(model.predict(features.iloc[[-1]][EXPECTED_FEATURES])[0])


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "solana_model_data.parquet",
    model_path: Path = MODELS_DIR / "solana_next_day_high.joblib",
    predictions_path: Path = PROCESSED_DATA_DIR / "solana_latest_prediction.csv",
) -> None:
    """Predict the next-day high using the final row of a history dataset."""
    model = joblib.load(model_path)
    raw_history = pd.read_parquet(input_path)
    prediction = predict_next_high(model, raw_history)

    output = pd.DataFrame([{"token": "SOL", "predicted_high": prediction}])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions_path, index=False)
    typer.echo(f"Wrote prediction to {predictions_path}: ${prediction:.4f}")


if __name__ == "__main__":
    app()

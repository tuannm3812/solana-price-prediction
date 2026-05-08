from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer

from solana_price_prediction.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Create reporting plots for Solana model outputs.")


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "solana_predictions.parquet",
    output_path: Path = FIGURES_DIR / "solana_actual_vs_predicted.png",
) -> None:
    """Plot actual vs predicted high prices from the test set."""
    results = pd.read_parquet(input_path)
    x_values = results["timestamp"] if "timestamp" in results.columns else results.index

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 7))
    plt.plot(x_values, results["target_next_day_high"], label="Actual high", alpha=0.75)
    plt.plot(x_values, results["predicted_high"], label="Predicted high", linestyle="--")
    plt.title("Solana Next-Day High: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    typer.echo(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    app()

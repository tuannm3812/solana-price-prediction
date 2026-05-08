import pandas as pd

from solana_price_prediction.dataset import load_solana_data_from_url
from solana_price_prediction.features import EXPECTED_FEATURES, SolanaFeatureEngineer, add_features, create_target_variable
from solana_price_prediction.modeling.train import split_time_series


def sample_history(rows: int = 40) -> pd.DataFrame:
    values = list(range(rows))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "open": [100 + value for value in values],
            "high": [105 + value for value in values],
            "low": [95 + value for value in values],
            "close": [102 + value for value in values],
            "volume": [1000 + value for value in values],
            "marketcap": [1_000_000 + value for value in values],
        }
    )


def test_create_target_variable_uses_next_day_high() -> None:
    df = sample_history(5)
    target_df = create_target_variable(df)

    assert len(target_df) == 4
    assert target_df.loc[0, "target_next_day_high"] == df.loc[1, "high"]


def test_feature_engineer_returns_expected_inference_columns() -> None:
    features = SolanaFeatureEngineer().transform(sample_history())

    assert list(features.columns) == EXPECTED_FEATURES
    assert len(features) == 40
    assert not features.iloc[[-1]].isna().any().any()


def test_training_features_drop_initial_rolling_rows() -> None:
    feature_df = add_features(create_target_variable(sample_history()))

    assert "sma_30" in feature_df.columns
    assert feature_df["target_next_day_high"].notna().all()
    assert len(feature_df) < 40


def test_time_series_split_preserves_order() -> None:
    df = sample_history(10)
    train_df, validation_df, test_df = split_time_series(df, validation_size=0.2, test_size=0.2)

    assert len(train_df) == 6
    assert len(validation_df) == 2
    assert len(test_df) == 2
    assert train_df["timestamp"].max() < validation_df["timestamp"].min()
    assert validation_df["timestamp"].max() < test_df["timestamp"].min()


def test_load_solana_data_from_csv_url(monkeypatch) -> None:
    class FakeResponse:
        content = (
            b"timestamp,open,high,low,close,volume,marketcap\n"
            b"2024-01-01,100,105,95,102,1000,1000000\n"
        )

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int):
        assert url == "https://example.com/solana.csv"
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr("solana_price_prediction.dataset.requests.get", fake_get)

    df = load_solana_data_from_url("https://example.com/solana.csv")

    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "marketcap"]
    assert df.loc[0, "high"] == 105

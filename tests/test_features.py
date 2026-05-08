import pandas as pd
from xgboost import XGBRegressor

from solana_price_prediction.dataset import load_solana_data, load_solana_data_from_url
from solana_price_prediction.features import (
    EXPECTED_FEATURES,
    SolanaFeatureEngineer,
    add_features,
    create_target_variable,
    select_feature_columns,
)
from solana_price_prediction.kraken import parse_kraken_ohlc
from solana_price_prediction.modeling.estimators import DeltaHighRegressor, PersistenceHighRegressor
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


def test_training_feature_selection_matches_inference_contract() -> None:
    feature_df = add_features(create_target_variable(sample_history()))
    feature_df["vwap"] = feature_df["close"]
    feature_df["count"] = 100

    assert select_feature_columns(feature_df) == EXPECTED_FEATURES


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


def test_parse_kraken_ohlc_payload() -> None:
    payload = {
        "error": [],
        "result": {
            "SOLUSD": [
                [1704067200, "100", "105", "95", "102", "101", "1000", 42],
                [1704153600, "102", "108", "101", "106", "104", "1200", 50],
            ],
            "last": 1704153600,
        },
    }

    df = parse_kraken_ohlc(payload)

    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "count",
        "marketcap",
    ]
    assert df.loc[0, "high"] == 105
    assert df.loc[1, "volume"] == 1200
    assert df.loc[0, "marketcap"] == 0.0


def test_load_solana_data_from_kraken(monkeypatch) -> None:
    def fake_since_timestamp(lookback_days: int) -> int:
        assert lookback_days == 30
        return 1704067200

    def fake_fetch_kraken_ohlc(pair: str, interval: int, since_timestamp: int):
        assert pair == "SOLUSD"
        assert interval == 1440
        assert since_timestamp == 1704067200
        return sample_history(2)

    monkeypatch.setattr("solana_price_prediction.dataset.kraken_since_timestamp", fake_since_timestamp)
    monkeypatch.setattr("solana_price_prediction.dataset.fetch_kraken_ohlc", fake_fetch_kraken_ohlc)

    df = load_solana_data(kraken_pair="SOLUSD", kraken_interval=1440, kraken_lookback_days=30)

    assert len(df) == 2
    assert df.loc[0, "close"] == 102


def test_delta_high_regressor_predicts_high_plus_residual() -> None:
    X = pd.DataFrame({"high": [100.0, 110.0, 120.0], "close": [99.0, 108.0, 119.0]})
    y = pd.Series([102.0, 112.0, 122.0])
    base = XGBRegressor(n_estimators=5, max_depth=1, random_state=42)
    model = DeltaHighRegressor(base).fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == 3
    assert list(model.feature_names_in_) == ["high", "close"]


def test_delta_high_regressor_can_clip_residuals() -> None:
    X = pd.DataFrame({"high": [100.0, 110.0, 120.0], "close": [99.0, 108.0, 119.0]})
    y = pd.Series([150.0, 160.0, 170.0])
    base = XGBRegressor(n_estimators=5, max_depth=1, random_state=42)
    model = DeltaHighRegressor(base, residual_clip=5.0).fit(X, y)

    predictions = model.predict(X)

    assert (predictions <= X["high"].to_numpy() + 5.0).all()
    assert (predictions >= X["high"].to_numpy() - 5.0).all()


def test_persistence_high_regressor_uses_current_high() -> None:
    X = pd.DataFrame({"high": [100.0, 110.0], "close": [99.0, 108.0]})
    model = PersistenceHighRegressor().fit(X)

    assert list(model.predict(X)) == [100.0, 110.0]

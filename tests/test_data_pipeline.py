from __future__ import annotations

import pandas as pd

from data_pipeline.cleaning import build_holiday_store_features, build_train_grid, clean_oil
from data_pipeline.features import build_feature_dataset, build_integrated_dataset


def test_build_train_grid_fills_missing_store_family_dates() -> None:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": pd.to_datetime(["2013-01-01", "2013-01-03", "2013-01-01"]),
            "store_nbr": [1, 1, 2],
            "family": ["A", "A", "A"],
            "sales": [10.0, 20.0, 5.0],
            "onpromotion": [1, 2, 0],
        }
    )

    filled, meta = build_train_grid(train)

    assert meta["rows_added"] == 3
    assert len(filled) == 6
    row = filled.loc[
        (filled["date"] == pd.Timestamp("2013-01-02"))
        & (filled["store_nbr"] == 1)
        & (filled["family"] == "A")
    ].iloc[0]
    assert row["sales"] == 0.0
    assert row["onpromotion"] == 0
    assert row["is_generated_row"] == 1


def test_clean_oil_interpolates_missing_dates() -> None:
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-03"]),
            "dcoilwtico": [90.0, 96.0],
        }
    )

    cleaned, meta = clean_oil(oil, pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-03"))

    assert meta["missing_original"] == 1
    middle = cleaned.loc[cleaned["date"] == pd.Timestamp("2013-01-02"), "oil_price"].iloc[0]
    assert middle == 93.0


def test_build_holiday_features_matches_store_scope() -> None:
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"]),
            "holiday_locale": ["national", "regional", "local"],
            "locale_name": ["ecuador", "pichincha", "quito"],
            "holiday_description": ["new year", "regional day", "city day"],
            "is_holiday": [1, 1, 1],
            "is_event": [0, 0, 0],
            "is_additional": [0, 0, 0],
            "is_bridge": [0, 0, 0],
            "is_work_day": [0, 0, 0],
        }
    )
    stores = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["quito", "guayaquil"],
            "state": ["pichincha", "guayas"],
        }
    )

    features = build_holiday_store_features(holidays, stores)

    national_hit = features.loc[
        (features["date"] == pd.Timestamp("2013-01-01")) & (features["store_nbr"] == 2)
    ].iloc[0]
    regional_hit = features.loc[
        (features["date"] == pd.Timestamp("2013-01-02")) & (features["store_nbr"] == 1)
    ].iloc[0]

    assert national_hit["is_national_holiday"] == 1
    assert regional_hit["is_regional_holiday"] == 1
    assert "regional day" in regional_hit["holiday_names"]


def test_feature_dataset_contains_expected_derived_columns() -> None:
    cleaned = {
        "train": pd.DataFrame(
            {
                "date": pd.to_datetime(["2013-01-01"]),
                "store_nbr": [1],
                "family": ["A"],
                "id": [1],
                "sales": [5.0],
                "onpromotion": [2],
                "is_generated_row": [0],
            }
        ),
        "test": pd.DataFrame(
            {
                "id": [2],
                "date": pd.to_datetime(["2013-01-02"]),
                "store_nbr": [1],
                "family": ["A"],
                "onpromotion": [0],
            }
        ),
        "stores": pd.DataFrame(
            {
                "store_nbr": [1],
                "city": ["quito"],
                "state": ["pichincha"],
                "type": ["d"],
                "cluster": [13],
            }
        ),
        "oil": pd.DataFrame(
            {
                "date": pd.to_datetime(["2013-01-01", "2013-01-02"]),
                "oil_price": [90.0, 91.0],
                "oil_missing_original": [0, 0],
                "oil_price_change_1": [0.0, 1.0],
                "oil_price_ma_7": [90.0, 90.5],
                "oil_price_ma_28": [90.0, 90.5],
            }
        ),
        "transactions": pd.DataFrame(
            {
                "date": pd.to_datetime(["2013-01-01", "2013-01-02"]),
                "store_nbr": [1, 1],
                "transactions": [100.0, 80.0],
                "transactions_missing_original": [0, 0],
            }
        ),
        "holiday_features": pd.DataFrame(
            {
                "date": pd.to_datetime(["2013-01-01"]),
                "store_nbr": [1],
                "holiday_count": [1],
                "is_national_holiday": [1],
                "is_regional_holiday": [0],
                "is_local_holiday": [0],
                "is_holiday": [1],
                "is_event": [0],
                "is_additional": [0],
                "is_bridge": [0],
                "is_work_day": [0],
                "holiday_names": ["new year"],
            }
        ),
    }

    integrated = build_integrated_dataset(cleaned)
    features = build_feature_dataset(integrated)

    assert {"promo_active", "is_weekend", "family_mean_transactions", "holiday_name_count"}.issubset(
        set(features.columns)
    )
    assert len(features) == 2

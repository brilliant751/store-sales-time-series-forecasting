from __future__ import annotations

import pandas as pd


JOIN_KEYS = ["date", "store_nbr", "family"]


def build_integrated_dataset(cleaned_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    train = cleaned_tables["train"].copy()
    test = cleaned_tables["test"].copy()

    train["dataset_split"] = "train"
    test["dataset_split"] = "test"
    test["sales"] = pd.NA
    test["is_generated_row"] = 0

    base = pd.concat(
        [train[JOIN_KEYS + ["id", "sales", "onpromotion", "dataset_split", "is_generated_row"]], test],
        ignore_index=True,
        sort=False,
    )

    stores = cleaned_tables["stores"].copy()
    oil = cleaned_tables["oil"].copy()
    transactions = cleaned_tables["transactions"].copy()
    holiday_features = cleaned_tables["holiday_features"].copy()

    merged = (
        base.merge(stores, on="store_nbr", how="left")
        .merge(oil, on="date", how="left")
        .merge(transactions, on=["date", "store_nbr"], how="left")
        .merge(holiday_features, on=["date", "store_nbr"], how="left")
        .sort_values(JOIN_KEYS)
        .reset_index(drop=True)
    )

    fill_zero_cols = [
        "holiday_count",
        "is_national_holiday",
        "is_regional_holiday",
        "is_local_holiday",
        "is_holiday",
        "is_event",
        "is_additional",
        "is_bridge",
        "is_work_day",
        "transactions_missing_original",
    ]
    for col in fill_zero_cols:
        merged[col] = merged[col].fillna(0).astype(int)
    merged["holiday_names"] = merged["holiday_names"].fillna("")
    merged["transactions"] = merged["transactions"].fillna(0.0)
    merged["promo_active"] = merged["onpromotion"].fillna(0).gt(0).astype(int)

    return merged


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["year"] = featured["date"].dt.year
    featured["quarter"] = featured["date"].dt.quarter
    featured["month"] = featured["date"].dt.month
    featured["day"] = featured["date"].dt.day
    featured["day_of_week"] = featured["date"].dt.dayofweek
    featured["day_of_year"] = featured["date"].dt.dayofyear
    featured["week_of_year"] = featured["date"].dt.isocalendar().week.astype(int)
    featured["is_weekend"] = featured["day_of_week"].isin([5, 6]).astype(int)
    featured["is_month_start"] = featured["date"].dt.is_month_start.astype(int)
    featured["is_month_end"] = featured["date"].dt.is_month_end.astype(int)
    featured["is_quarter_start"] = featured["date"].dt.is_quarter_start.astype(int)
    featured["is_quarter_end"] = featured["date"].dt.is_quarter_end.astype(int)
    featured["days_since_start"] = (featured["date"] - featured["date"].min()).dt.days
    return featured


def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured = featured.sort_values(JOIN_KEYS).reset_index(drop=True)

    featured["store_family_key"] = (
        featured["store_nbr"].astype(str) + "_" + featured["family"].astype(str)
    )
    featured["family_mean_onpromotion"] = featured.groupby("family")["onpromotion"].transform("mean")
    featured["store_mean_onpromotion"] = featured.groupby("store_nbr")["onpromotion"].transform("mean")
    featured["family_mean_transactions"] = featured.groupby("family")["transactions"].transform("mean")
    featured["store_mean_transactions"] = featured.groupby("store_nbr")["transactions"].transform("mean")
    featured["holiday_name_count"] = featured["holiday_names"].map(lambda x: 0 if not x else len(x.split("|")))
    return featured


def build_feature_dataset(integrated_data: pd.DataFrame) -> pd.DataFrame:
    featured = add_calendar_features(integrated_data)
    featured = add_group_features(featured)

    ordered_columns = [
        "id",
        "date",
        "dataset_split",
        "store_nbr",
        "family",
        "store_family_key",
        "sales",
        "onpromotion",
        "promo_active",
        "transactions",
        "transactions_missing_original",
        "oil_price",
        "oil_missing_original",
        "oil_price_change_1",
        "oil_price_ma_7",
        "oil_price_ma_28",
        "city",
        "state",
        "type",
        "cluster",
        "holiday_count",
        "holiday_name_count",
        "holiday_names",
        "is_national_holiday",
        "is_regional_holiday",
        "is_local_holiday",
        "is_holiday",
        "is_event",
        "is_additional",
        "is_bridge",
        "is_work_day",
        "year",
        "quarter",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "days_since_start",
        "family_mean_onpromotion",
        "store_mean_onpromotion",
        "family_mean_transactions",
        "store_mean_transactions",
        "is_generated_row",
    ]
    return featured[ordered_columns].sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)

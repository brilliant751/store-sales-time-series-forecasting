from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


TRAIN_KEY = ["date", "store_nbr", "family"]


def normalize_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _ensure_types(raw_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    train = raw_tables["train"].copy()
    test = raw_tables["test"].copy()
    stores = raw_tables["stores"].copy()
    oil = raw_tables["oil"].copy()
    holidays = raw_tables["holidays"].copy()
    transactions = raw_tables["transactions"].copy()

    for frame in (train, test, stores, transactions):
        if "store_nbr" in frame.columns:
            frame["store_nbr"] = frame["store_nbr"].astype(int)

    train["family"] = train["family"].astype(str)
    test["family"] = test["family"].astype(str)
    train["sales"] = train["sales"].astype(float)
    train["onpromotion"] = train["onpromotion"].astype(int)
    test["onpromotion"] = test["onpromotion"].astype(int)
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
    transactions["transactions"] = pd.to_numeric(transactions["transactions"], errors="coerce")
    holidays["transferred"] = holidays["transferred"].astype(str).str.lower().eq("true")

    for col in ("city", "state", "type"):
        stores[col] = stores[col].map(normalize_text)
    stores["cluster"] = stores["cluster"].astype(int)

    for col in ("type", "locale", "locale_name", "description"):
        holidays[col] = holidays[col].map(normalize_text)

    return {
        "train": train,
        "test": test,
        "stores": stores,
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
    }


def build_train_grid(train: pd.DataFrame, fill_value: float = 0.0) -> tuple[pd.DataFrame, dict[str, int]]:
    min_date = train["date"].min()
    max_date = train["date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D")
    stores = np.sort(train["store_nbr"].unique())
    families = np.sort(train["family"].unique())

    full_index = pd.MultiIndex.from_product(
        [all_dates, stores, families],
        names=TRAIN_KEY,
    )
    grid = pd.DataFrame(index=full_index).reset_index()
    merged = grid.merge(train, on=TRAIN_KEY, how="left", indicator=True)
    merged["is_generated_row"] = merged["_merge"].eq("left_only").astype(int)
    merged = merged.drop(columns="_merge")
    merged["id"] = merged["id"].astype("Int64")
    merged["sales"] = merged["sales"].fillna(fill_value)
    merged["onpromotion"] = merged["onpromotion"].fillna(0).astype(int)

    metadata = {
        "rows_before": int(len(train)),
        "rows_after": int(len(merged)),
        "rows_added": int(merged["is_generated_row"].sum()),
    }
    return merged.sort_values(TRAIN_KEY).reset_index(drop=True), metadata


def clean_oil(oil: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, int]]:
    base_dates = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D")})
    merged = base_dates.merge(oil.rename(columns={"dcoilwtico": "oil_price"}), on="date", how="left")
    merged["oil_missing_original"] = merged["oil_price"].isna().astype(int)
    merged["oil_price"] = merged["oil_price"].interpolate(method="linear", limit_direction="both")
    merged["oil_price"] = merged["oil_price"].ffill().bfill()
    merged["oil_price_change_1"] = merged["oil_price"].diff().fillna(0.0)
    merged["oil_price_ma_7"] = merged["oil_price"].rolling(window=7, min_periods=1).mean()
    merged["oil_price_ma_28"] = merged["oil_price"].rolling(window=28, min_periods=1).mean()

    metadata = {
        "rows_after_reindex": int(len(merged)),
        "missing_original": int(merged["oil_missing_original"].sum()),
    }
    return merged, metadata


def clean_holidays(holidays: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    original_rows = len(holidays)
    filtered = holidays.loc[~holidays["transferred"]].copy()
    filtered["holiday_type"] = filtered["type"]
    filtered["holiday_locale"] = filtered["locale"]
    filtered["holiday_description"] = filtered["description"]
    filtered["is_holiday"] = filtered["holiday_type"].eq("holiday").astype(int)
    filtered["is_event"] = filtered["holiday_type"].eq("event").astype(int)
    filtered["is_additional"] = filtered["holiday_type"].eq("additional").astype(int)
    filtered["is_bridge"] = filtered["holiday_type"].eq("bridge").astype(int)
    filtered["is_work_day"] = filtered["holiday_type"].eq("work day").astype(int)

    metadata = {
        "rows_before": int(original_rows),
        "rows_after": int(len(filtered)),
        "transferred_removed": int(original_rows - len(filtered)),
    }
    return filtered, metadata


def build_transactions_frame(
    transactions: pd.DataFrame,
    stores: pd.DataFrame,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    skeleton = pd.MultiIndex.from_product(
        [pd.date_range(min_date, max_date, freq="D"), np.sort(stores["store_nbr"].unique())],
        names=["date", "store_nbr"],
    )
    frame = pd.DataFrame(index=skeleton).reset_index()
    merged = frame.merge(transactions, on=["date", "store_nbr"], how="left")
    merged["transactions_missing_original"] = merged["transactions"].isna().astype(int)
    merged["transactions"] = merged["transactions"].fillna(0.0)
    metadata = {
        "rows_after_reindex": int(len(merged)),
        "missing_original": int(merged["transactions_missing_original"].sum()),
    }
    return merged, metadata


def build_holiday_store_features(
    holidays: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Expand holidays to store level and aggregate store-date features."""

    national = holidays.loc[holidays["holiday_locale"].eq("national")].copy()
    national["join_key"] = 1
    stores_all = stores[["store_nbr", "city", "state"]].copy()
    stores_all["join_key"] = 1
    national_expanded = national.merge(stores_all, on="join_key", how="left").drop(columns="join_key")

    regional = holidays.loc[holidays["holiday_locale"].eq("regional")].copy()
    regional = regional.merge(stores[["store_nbr", "city", "state"]], left_on="locale_name", right_on="state", how="left")

    local = holidays.loc[holidays["holiday_locale"].eq("local")].copy()
    local = local.merge(stores[["store_nbr", "city", "state"]], left_on="locale_name", right_on="city", how="left")

    combined = pd.concat([national_expanded, regional, local], ignore_index=True)
    combined = combined.dropna(subset=["store_nbr"]).copy()
    combined["store_nbr"] = combined["store_nbr"].astype(int)
    combined["holiday_name"] = combined["holiday_description"].fillna("")

    aggregated = (
        combined.groupby(["date", "store_nbr"], as_index=False)
        .agg(
            holiday_count=("holiday_name", "size"),
            is_national_holiday=("holiday_locale", lambda s: int((s == "national").any())),
            is_regional_holiday=("holiday_locale", lambda s: int((s == "regional").any())),
            is_local_holiday=("holiday_locale", lambda s: int((s == "local").any())),
            is_holiday=("is_holiday", "max"),
            is_event=("is_event", "max"),
            is_additional=("is_additional", "max"),
            is_bridge=("is_bridge", "max"),
            is_work_day=("is_work_day", "max"),
            holiday_names=("holiday_name", lambda s: "|".join(sorted({x for x in s if x}))),
        )
    )
    return aggregated


def standardize_tables(raw_tables: dict[str, pd.DataFrame], fill_value: float = 0.0) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    typed = _ensure_types(raw_tables)
    train, train_meta = build_train_grid(typed["train"], fill_value=fill_value)

    min_date = min(train["date"].min(), typed["test"]["date"].min())
    max_date = max(train["date"].max(), typed["test"]["date"].max())

    oil, oil_meta = clean_oil(typed["oil"], min_date=min_date, max_date=max_date)
    holidays, holiday_meta = clean_holidays(typed["holidays"])
    transactions, transaction_meta = build_transactions_frame(
        typed["transactions"],
        typed["stores"],
        min_date=min_date,
        max_date=max_date,
    )
    holiday_features = build_holiday_store_features(holidays, typed["stores"])

    cleaned = {
        "train": train,
        "test": typed["test"].copy(),
        "stores": typed["stores"].copy(),
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
        "holiday_features": holiday_features,
    }
    metadata = {
        "train": train_meta,
        "oil": oil_meta,
        "holidays": holiday_meta,
        "transactions": transaction_meta,
    }
    return cleaned, metadata

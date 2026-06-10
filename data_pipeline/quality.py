from __future__ import annotations

from typing import Any

import pandas as pd


def profile_table(df: pd.DataFrame, keys: list[str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_by_column": {col: int(val) for col, val in df.isna().sum().to_dict().items()},
    }
    if keys:
        summary["duplicate_key_rows"] = int(df.duplicated(keys).sum())
    if "date" in df.columns and not df.empty:
        summary["date_min"] = df["date"].min().strftime("%Y-%m-%d")
        summary["date_max"] = df["date"].max().strftime("%Y-%m-%d")
    return summary


def build_quality_report(
    raw_tables: dict[str, pd.DataFrame],
    cleaned_tables: dict[str, pd.DataFrame],
    transformation_metadata: dict[str, Any],
    integrated_data: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "raw_tables": {
            "train": profile_table(raw_tables["train"], keys=["id"]),
            "test": profile_table(raw_tables["test"], keys=["id"]),
            "stores": profile_table(raw_tables["stores"], keys=["store_nbr"]),
            "oil": profile_table(raw_tables["oil"], keys=["date"]),
            "holidays": profile_table(raw_tables["holidays"]),
            "transactions": profile_table(raw_tables["transactions"], keys=["date", "store_nbr"]),
        },
        "cleaned_tables": {
            "train": profile_table(cleaned_tables["train"], keys=["date", "store_nbr", "family"]),
            "test": profile_table(cleaned_tables["test"], keys=["id"]),
            "oil": profile_table(cleaned_tables["oil"], keys=["date"]),
            "holidays": profile_table(cleaned_tables["holidays"]),
            "transactions": profile_table(cleaned_tables["transactions"], keys=["date", "store_nbr"]),
            "holiday_features": profile_table(cleaned_tables["holiday_features"], keys=["date", "store_nbr"]),
        },
        "transformations": transformation_metadata,
        "outputs": {
            "integrated_dataset": profile_table(integrated_data, keys=["date", "store_nbr", "family", "dataset_split"]),
            "feature_dataset": profile_table(features, keys=["date", "store_nbr", "family", "dataset_split"]),
        },
    }

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig


def load_raw_tables(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Load all source tables with consistent parsing rules."""

    return {
        "train": pd.read_csv(config.train_path, parse_dates=["date"]),
        "test": pd.read_csv(config.test_path, parse_dates=["date"]),
        "stores": pd.read_csv(config.stores_path),
        "oil": pd.read_csv(config.oil_path, parse_dates=["date"]),
        "holidays": pd.read_csv(config.holidays_path, parse_dates=["date"]),
        "transactions": pd.read_csv(config.transactions_path, parse_dates=["date"]),
    }


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

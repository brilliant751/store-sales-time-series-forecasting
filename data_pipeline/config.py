from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for raw inputs and generated artifacts."""

    raw_data_dir: Path
    output_dir: Path
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    stores_file: str = "stores.csv"
    oil_file: str = "oil.csv"
    holidays_file: str = "holidays_events.csv"
    transactions_file: str = "transactions.csv"
    full_grid_fill_value: float = 0.0

    @property
    def train_path(self) -> Path:
        return self.raw_data_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.raw_data_dir / self.test_file

    @property
    def stores_path(self) -> Path:
        return self.raw_data_dir / self.stores_file

    @property
    def oil_path(self) -> Path:
        return self.raw_data_dir / self.oil_file

    @property
    def holidays_path(self) -> Path:
        return self.raw_data_dir / self.holidays_file

    @property
    def transactions_path(self) -> Path:
        return self.raw_data_dir / self.transactions_file

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

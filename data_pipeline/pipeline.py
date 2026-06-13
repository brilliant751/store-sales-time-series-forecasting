from __future__ import annotations

from pathlib import Path
from typing import Any

from .cleaning import standardize_tables
from .config import PipelineConfig
from .features import build_feature_dataset, build_integrated_dataset
from .io_utils import load_raw_tables, write_dataframe, write_json
from .lineage import build_lineage_metadata
from .quality import build_quality_report


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    output_dir = config.ensure_output_dir()
    raw_tables = load_raw_tables(config)
    cleaned_tables, transformation_metadata = standardize_tables(
        raw_tables,
        fill_value=config.full_grid_fill_value,
    )

    integrated_data = build_integrated_dataset(cleaned_tables)
    feature_data = build_feature_dataset(integrated_data)
    quality_report = build_quality_report(
        raw_tables=raw_tables,
        cleaned_tables=cleaned_tables,
        transformation_metadata=transformation_metadata,
        integrated_data=integrated_data,
        features=feature_data,
    )
    lineage_metadata = build_lineage_metadata(config.raw_data_dir, output_dir)

    write_dataframe(integrated_data, output_dir / "integrated_dataset.csv")
    write_dataframe(feature_data, output_dir / "feature_dataset.csv")
    write_json(quality_report, output_dir / "quality_report.json")
    write_json(lineage_metadata, output_dir / "lineage_metadata.json")

    return {
        "integrated_dataset": output_dir / "integrated_dataset.csv",
        "feature_dataset": output_dir / "feature_dataset.csv",
        "quality_report": output_dir / "quality_report.json",
        "lineage_metadata": output_dir / "lineage_metadata.json",
    }


def default_config(project_root: Path) -> PipelineConfig:
    return PipelineConfig(
        raw_data_dir=project_root / "data",
        output_dir=project_root / "data" / "processed",
    )

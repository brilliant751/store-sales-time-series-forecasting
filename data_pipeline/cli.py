from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import default_config, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the store-sales data pipeline.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing the raw CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated datasets and reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    base_config = default_config(project_root)

    config = PipelineConfig(
        raw_data_dir=args.raw_dir or base_config.raw_data_dir,
        output_dir=args.output_dir or base_config.output_dir,
    )
    outputs = run_pipeline(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.config import PipelineConfig
from data_pipeline.pipeline import run_pipeline
from src.models.baseline import SeasonalNaive
from src.models.utils import (
    evaluate_all,
    load_sequence_list,
    load_time_series,
    log_experiment,
    rolling_origin_split,
    time_split,
)

warnings.filterwarnings("ignore")


def run_data_pipeline(
    project_root: Path,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> Path:
    raw_data_dir = raw_dir or (project_root / "data")
    output_dir = processed_dir or (project_root / "data" / "processed")
    config = PipelineConfig(raw_data_dir=raw_data_dir, output_dir=output_dir)

    print("Running data pipeline...")
    outputs = run_pipeline(config)
    feature_dataset = outputs["feature_dataset"]
    print(f"  feature_dataset: {feature_dataset}")
    return feature_dataset


def run_baseline_all(
    data_dir: Path,
    n_sequences: int | None = None,
    output_csv: str = "output/baseline_results.csv",
) -> pd.DataFrame:
    pairs = load_sequence_list(data_dir=str(data_dir))
    if n_sequences:
        pairs = pairs[:n_sequences]

    print(f"Running Seasonal Naive on {len(pairs)} sequences...")
    results = []
    t0 = time.time()

    for idx, (store, family) in enumerate(pairs):
        try:
            series = load_time_series(store, family, cols=["sales"], data_dir=str(data_dir))
            train, val, test = time_split(series)
            if train.empty or val.empty:
                print(f"  [{idx + 1}/{len(pairs)}] SKIP ({store}, {family}): insufficient split")
                continue

            model = SeasonalNaive(period=7)
            model.fit(train["sales"])
            preds = model.predict(len(val))
            metrics = evaluate_all(val["sales"].values, preds)

            results.append(
                {
                    "store_nbr": store,
                    "family": family,
                    "rmsle": metrics["RMSLE"],
                    "rmse": metrics["RMSE"],
                    "mae": metrics["MAE"],
                    "train_days": len(train),
                    "val_days": len(val),
                    "test_days": len(test),
                }
            )

            if (idx + 1) % 100 == 0:
                recent = [r["rmsle"] for r in results[-100:]]
                print(f"  [{idx + 1}/{len(pairs)}] RMSLE avg={np.mean(recent):.4f}")
        except Exception as e:
            print(f"  [{idx + 1}/{len(pairs)}] SKIP ({store}, {family}): {e}")

    elapsed = time.time() - t0
    df = pd.DataFrame(results)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if df.empty:
        print("\nNo valid sequences were evaluated.")
        return df

    summary = {
        "n_sequences": len(df),
        "rmsle_mean": df["rmsle"].mean(),
        "rmsle_median": df["rmsle"].median(),
        "rmsle_std": df["rmsle"].std(),
        "rmse_mean": df["rmse"].mean(),
        "mae_mean": df["mae"].mean(),
    }

    print(f"\n=== Baseline Summary ({len(df)} sequences) ===")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"  runtime: {elapsed:.1f}s")
    print(f"  output: {output_csv}")

    log_experiment(
        experiment_name="baseline_all_sequences",
        model_name="SeasonalNaive",
        params={"period": 7},
        metrics={"rmsle_mean": summary["rmsle_mean"]},
        runtime_seconds=elapsed,
        extra={
            "n_sequences": summary["n_sequences"],
            "output_csv": output_csv,
            "metric_details": output_csv,
        },
    )
    return df


def run_rolling_validation(
    data_dir: Path,
    store_nbr: int = 1,
    family: str = "GROCERY I",
    n_splits: int = 3,
    output_csv: str | None = None,
) -> pd.DataFrame:
    series = load_time_series(store_nbr, family, cols=["sales"], data_dir=str(data_dir))
    results = []
    t0 = time.time()

    for train, val, split_idx in rolling_origin_split(series, n_splits=n_splits):
        model = SeasonalNaive(period=7)
        model.fit(train["sales"])
        preds = model.predict(len(val))
        metrics = evaluate_all(val["sales"].values, preds)
        results.append(
            {
                "split": split_idx,
                "train_end": train.index[-1].strftime("%Y-%m-%d"),
                "val_start": val.index[0].strftime("%Y-%m-%d"),
                "val_end": val.index[-1].strftime("%Y-%m-%d"),
                **metrics,
            }
        )
        print(
            f"  Split {split_idx}: val={val.index[0].strftime('%Y-%m-%d')}~{val.index[-1].strftime('%Y-%m-%d')} "
            f"RMSLE={metrics['RMSLE']:.4f}"
        )

    elapsed = time.time() - t0
    df = pd.DataFrame(results)
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    log_experiment(
        experiment_name="rolling_validation",
        model_name="SeasonalNaive",
        params={"period": 7, "n_splits": n_splits, "store_nbr": store_nbr, "family": family},
        metrics={"rmsle_mean": df["RMSLE"].mean()},
        runtime_seconds=elapsed,
        extra={"details": output_csv},
    )
    print(f"  Rolling avg RMSLE: {df['RMSLE'].mean():.4f}  (runtime: {elapsed:.1f}s)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data pipeline and train baseline models.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory containing raw CSV files.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory for generated datasets and reports.",
    )
    parser.add_argument("--n-sequences", type=int, default=None, help="Number of sequences to process.")
    parser.add_argument("--rolling", action="store_true", help="Run rolling validation on one sequence.")
    parser.add_argument("--store", type=int, default=1, help="Store for rolling validation.")
    parser.add_argument("--family", type=str, default="GROCERY I", help="Family for rolling validation.")
    parser.add_argument("--output", default="output/baseline_results.csv", help="Output CSV for summary.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip data pipeline and reuse existing features.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = args.processed_dir or (project_root / "data" / "processed")

    if not args.skip_pipeline:
        run_data_pipeline(project_root, raw_dir=args.raw_dir, processed_dir=processed_dir)

    feature_dir = processed_dir
    feature_path = feature_dir / "feature_dataset.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"feature dataset not found: {feature_path}")

    if args.rolling:
        run_rolling_validation(
            feature_dir,
            store_nbr=args.store,
            family=args.family,
        )
    else:
        run_baseline_all(
            feature_dir,
            n_sequences=args.n_sequences,
            output_csv=args.output,
        )

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

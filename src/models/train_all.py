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
from src.models.utils import evaluate_all, log_experiment, rolling_origin_split, time_split

warnings.filterwarnings("ignore")

TARGET_COLUMN = "sales"
SPLIT_COLUMN = "dataset_split"
VAL_START = pd.Timestamp("2017-07-01")
TEST_START = pd.Timestamp("2017-08-01")
FEATURE_EXCLUDE_COLUMNS = {
    "id",
    "date",
    "dataset_split",
    "sales",
    "holiday_names",
    "store_family_key",
}


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


def load_training_frame(data_dir: Path) -> pd.DataFrame:
    feature_path = data_dir / "feature_dataset.csv"
    print(f"Loading feature dataset once: {feature_path}")
    df = pd.read_csv(
        feature_path,
        usecols=["date", "store_nbr", "family", "sales", "dataset_split"],
    )
    df = df[df["dataset_split"] == "train"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["store_nbr"] = df["store_nbr"].astype(int)
    df["family"] = df["family"].astype(str)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    print(f"  loaded train rows: {len(df)}")
    return df


def load_feature_frame(data_dir: Path) -> pd.DataFrame:
    feature_path = data_dir / "feature_dataset.csv"
    print(f"Loading feature dataset for modeling: {feature_path}")
    df = pd.read_csv(feature_path)
    df["date"] = pd.to_datetime(df["date"])
    df["store_nbr"] = df["store_nbr"].astype(int)
    df["family"] = df["family"].astype(str)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    print(f"  loaded rows: {len(df)}")
    return df


def load_pairs_from_frame(df: pd.DataFrame) -> list[tuple[int, str]]:
    pairs = df[["store_nbr", "family"]].drop_duplicates().values.tolist()
    return [(int(store), str(family)) for store, family in pairs]


def slice_time_series(df: pd.DataFrame, store_nbr: int, family: str) -> pd.DataFrame:
    mask = (df["store_nbr"] == store_nbr) & (df["family"] == family)
    series = df.loc[mask, ["date", "sales"]].copy()
    return series.sort_values("date").set_index("date")


def filter_frame_to_pairs(df: pd.DataFrame, pairs: list[tuple[int, str]]) -> pd.DataFrame:
    pair_index = pd.MultiIndex.from_tuples(pairs, names=["store_nbr", "family"])
    row_index = pd.MultiIndex.from_frame(df[["store_nbr", "family"]])
    return df.loc[row_index.isin(pair_index)].copy()


def run_baseline_all(
    data_dir: Path,
    n_sequences: int | None = None,
    output_csv: str = "output/baseline_results.csv",
) -> pd.DataFrame:
    training_frame = load_training_frame(data_dir)
    pairs = load_pairs_from_frame(training_frame)
    if n_sequences:
        pairs = pairs[:n_sequences]

    print(f"Running Seasonal Naive on {len(pairs)} sequences...")
    results = []
    t0 = time.time()

    for idx, (store, family) in enumerate(pairs):
        try:
            series = slice_time_series(training_frame, store, family)
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
                    "RMSLE": metrics["RMSLE"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "train_days": len(train),
                    "val_days": len(val),
                    "test_days": len(test),
                }
            )

            if (idx + 1) % 100 == 0:
                recent = [r["RMSLE"] for r in results[-100:]]
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
        "rmsle_mean": df["RMSLE"].mean(),
        "rmsle_median": df["RMSLE"].median(),
        "rmsle_std": df["RMSLE"].std(),
        "rmse_mean": df["RMSE"].mean(),
        "mae_mean": df["MAE"].mean(),
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


def prepare_tabular_matrices(
    df: pd.DataFrame,
    val_start: pd.Timestamp = VAL_START,
    test_start: pd.Timestamp = TEST_START,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    train_rows = df[
        (df[SPLIT_COLUMN] == "train")
        & df[TARGET_COLUMN].notna()
        & (df["date"] < test_start)
    ].copy()
    train_rows = train_rows.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)

    feature_columns = [col for col in train_rows.columns if col not in FEATURE_EXCLUDE_COLUMNS]
    X = train_rows[feature_columns].copy()
    y = pd.to_numeric(train_rows[TARGET_COLUMN], errors="coerce").fillna(0.0)

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns, dummy_na=False)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    split_mask = train_rows["date"] < val_start
    X_train = X.loc[split_mask].reset_index(drop=True)
    y_train = y.loc[split_mask].reset_index(drop=True)
    X_val = X.loc[~split_mask].reset_index(drop=True)
    y_val = y.loc[~split_mask].reset_index(drop=True)

    if X_train.empty or X_val.empty:
        raise ValueError("train/validation split is empty; check date range.")

    return X_train, y_train, X_val, y_val, X.columns.tolist()


def _restrict_to_first_sequences(df: pd.DataFrame, n_sequences: int | None) -> pd.DataFrame:
    if not n_sequences:
        return df
    pairs = load_pairs_from_frame(df[df[SPLIT_COLUMN] == "train"])
    selected_pairs = pairs[:n_sequences]
    filtered = filter_frame_to_pairs(df, selected_pairs)
    print(f"  filtered to first {n_sequences} sequences: {len(filtered)} rows")
    return filtered


def _sample_sequences_for_deep_learning(df: pd.DataFrame, n_sequences: int | None, seed: int = 42) -> pd.DataFrame:
    if not n_sequences:
        return df

    train_pairs = load_pairs_from_frame(df[df[SPLIT_COLUMN] == "train"])
    if n_sequences >= len(train_pairs):
        return df

    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(train_pairs), size=n_sequences, replace=False))
    selected_pairs = [train_pairs[i] for i in selected_indices]
    filtered = filter_frame_to_pairs(df, selected_pairs)
    print(f"  sampled {n_sequences} sequences for deep learning: {len(filtered)} rows")
    return filtered


def run_xgboost_model(
    data_dir: Path,
    n_sequences: int | None = None,
    model_output_path: str = "outputs/models/xgboost_model.pkl",
    metrics_output_path: str = "output/xgboost_metrics.csv",
    feature_importance_path: str = "outputs/models/xgboost_feature_importance.csv",
    n_estimators: int = 300,
) -> dict[str, float]:
    from src.models.xgboost.model import XGBoostModel

    t0 = time.time()
    feature_frame = _restrict_to_first_sequences(load_feature_frame(data_dir), n_sequences)

    X_train, y_train, X_val, y_val, feature_names = prepare_tabular_matrices(feature_frame)
    print(f"Training XGBoost: train={X_train.shape}, val={X_val.shape}")

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 8,
        "n_estimators": n_estimators,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }
    model = XGBoostModel(params=params)
    model.train(X_train, y_train, X_val, y_val, verbose=50)

    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    elapsed = time.time() - t0

    model_path = Path(model_output_path)
    metrics_path = Path(metrics_output_path)
    importance_path = Path(feature_importance_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_path))
    model.get_feature_importance(feature_names).to_csv(importance_path, index=False)

    metrics_df = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "validation", **val_metrics},
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)

    print("\n=== XGBoost Metrics ===")
    print(metrics_df.to_string(index=False))
    print(f"  runtime: {elapsed:.1f}s")
    print(f"  model: {model_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  feature importance: {importance_path}")

    log_experiment(
        experiment_name="xgboost_feature_dataset",
        model_name="XGBoost",
        params=params,
        metrics={f"val_{key.lower()}": float(value) for key, value in val_metrics.items()},
        runtime_seconds=elapsed,
        extra={
            "n_sequences": n_sequences,
            "model_output_path": str(model_path),
            "metrics_output_path": str(metrics_path),
            "feature_importance_path": str(importance_path),
        },
    )
    return val_metrics


def run_deep_learning_model(
    data_dir: Path,
    model_type: str,
    n_sequences: int | None = None,
    model_output_path: str | None = None,
    metrics_output_path: str | None = None,
) -> dict[str, float]:
    from src.models.deep_learning.model import DeepLearningModel

    t0 = time.time()
    feature_frame = _sample_sequences_for_deep_learning(load_feature_frame(data_dir), n_sequences)
    X_train, y_train, X_val, y_val, _ = prepare_tabular_matrices(feature_frame)
    print(f"Training {model_type.upper()}: train={X_train.shape}, val={X_val.shape}")

    params = {
        "units": 64,
        "layers": 1,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "sequence_length": 14,
        "epochs": 10,
        "batch_size": 256,
    }
    model = DeepLearningModel(model_type=model_type, params=params)
    history = model.train(X_train, y_train, X_val, y_val)

    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    elapsed = time.time() - t0

    model_output_path = model_output_path or f"outputs/models/{model_type}_model.h5"
    metrics_output_path = metrics_output_path or f"output/{model_type}_metrics.csv"
    model_path = Path(model_output_path)
    metrics_path = Path(metrics_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_path))

    metrics_df = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "validation", **val_metrics},
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n=== {model_type.upper()} Metrics ===")
    print(metrics_df.to_string(index=False))
    print(f"  runtime: {elapsed:.1f}s")
    print(f"  model: {model_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  epochs_trained: {len(history.history.get('loss', []))}")

    log_experiment(
        experiment_name=f"{model_type}_feature_dataset",
        model_name=model_type.upper(),
        params=params,
        metrics={f"val_{key.lower()}": float(value) for key, value in val_metrics.items()},
        runtime_seconds=elapsed,
        extra={
            "n_sequences": n_sequences,
            "model_output_path": str(model_path),
            "metrics_output_path": str(metrics_path),
        },
    )
    return val_metrics


def run_rolling_validation(
    data_dir: Path,
    store_nbr: int = 1,
    family: str = "GROCERY I",
    n_splits: int = 3,
    output_csv: str | None = None,
) -> pd.DataFrame:
    training_frame = load_training_frame(data_dir)
    series = slice_time_series(training_frame, store_nbr, family)
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


def compare_models(metrics_history: dict[str, dict[str, float]]) -> None:
    if len(metrics_history) < 2:
        return
    from src.evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator()
    for model_name, metrics in metrics_history.items():
        evaluator.add_metrics(model_name, metrics)
    evaluator.plot_comparison()
    report_path = evaluator.generate_report()
    print(f"\nComparison report saved to {report_path}")
    best_model, best_metrics = evaluator.get_best_model()
    print(f"Best model: {best_model}")
    print(f"Best metrics: {best_metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data pipeline and train models.")
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
    parser.add_argument("--output", default="output/baseline_results.csv", help="Output CSV for baseline summary.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip data pipeline and reuse existing features.")
    parser.add_argument(
        "--model",
        choices=["baseline", "xgboost", "lstm", "gru", "all"],
        default="all",
        help="Model to train after the data pipeline. 'all' means xgboost+lstm+gru.",
    )
    parser.add_argument("--xgb-rounds", type=int, default=300, help="Number of XGBoost boosting rounds.")
    parser.add_argument("--dl-sequences", type=int, default=10, help="Number of store/family sequences for LSTM/GRU.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = args.processed_dir or (project_root / "data" / "processed")

    if not args.skip_pipeline:
        run_data_pipeline(project_root, raw_dir=args.raw_dir, processed_dir=processed_dir)

    feature_path = processed_dir / "feature_dataset.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"feature dataset not found: {feature_path}")

    metrics_history: dict[str, dict[str, float]] = {}

    if args.rolling:
        run_rolling_validation(
            processed_dir,
            store_nbr=args.store,
            family=args.family,
        )
        print("\nPipeline completed successfully!")
        return

    if args.model == "baseline":
        baseline_df = run_baseline_all(
            processed_dir,
            n_sequences=args.n_sequences,
            output_csv=args.output,
        )
        if not baseline_df.empty:
            metrics_history["SeasonalNaive"] = {
                "RMSLE": float(baseline_df["RMSLE"].mean()),
                "MAE": float(baseline_df["MAE"].mean()),
                "RMSE": float(baseline_df["RMSE"].mean()),
            }

    if args.model in ("xgboost", "all"):
        metrics_history["XGBoost"] = run_xgboost_model(
            processed_dir,
            n_sequences=args.n_sequences,
            n_estimators=args.xgb_rounds,
        )

    if args.model in ("lstm", "all"):
        metrics_history["LSTM"] = run_deep_learning_model(
            processed_dir,
            model_type="lstm",
            n_sequences=args.dl_sequences,
        )

    if args.model in ("gru", "all"):
        metrics_history["GRU"] = run_deep_learning_model(
            processed_dir,
            model_type="gru",
            n_sequences=args.dl_sequences,
        )

    compare_models(metrics_history)
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

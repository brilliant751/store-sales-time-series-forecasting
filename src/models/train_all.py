import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.xgboost.train import train_xgboost_model
from src.models.deep_learning.train import train_deep_learning_model
from src.evaluation.evaluator import ModelEvaluator

def main():
    print("Starting model training and evaluation pipeline...")
    
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    train_filepath = 'data/train.csv'
    
    if not os.path.exists(train_filepath):
        print(f"Error: Training data not found at {train_filepath}")
        print("Please place your train.csv in the data/ directory.")
        return
    
    evaluator = ModelEvaluator()
    
    print("\n=== Training XGBoost Model ===")
    try:
        xgb_model, xgb_train_metrics, xgb_val_metrics = train_xgboost_model(
            train_filepath,
            'outputs/models/xgboost_model.pkl',
            'logs/xgboost_training.log'
        )
        evaluator.add_metrics('XGBoost', xgb_val_metrics)
        print(f"XGBoost Validation Metrics: {xgb_val_metrics}")
    except Exception as e:
        print(f"Error training XGBoost: {e}")
    
    print("\n=== Training LSTM Model ===")
    try:
        lstm_model, lstm_train_metrics, lstm_val_metrics, _ = train_deep_learning_model(
            'lstm',
            train_filepath,
            'outputs/models/lstm_model.h5',
            'logs/lstm_training.log'
        )
        evaluator.add_metrics('LSTM', lstm_val_metrics)
        print(f"LSTM Validation Metrics: {lstm_val_metrics}")
    except Exception as e:
        print(f"Error training LSTM: {e}")
    
    print("\n=== Training GRU Model ===")
    try:
        gru_model, gru_train_metrics, gru_val_metrics, _ = train_deep_learning_model(
            'gru',
            train_filepath,
            'outputs/models/gru_model.h5',
            'logs/gru_training.log'
        )
        evaluator.add_metrics('GRU', gru_val_metrics)
        print(f"GRU Validation Metrics: {gru_val_metrics}")
    except Exception as e:
        print(f"Error training GRU: {e}")
    
    print("\n=== Generating Comparison Report ===")
    evaluator.plot_comparison()
    report = evaluator.generate_report()
    print("Comparison report generated.")
    
    best_model, best_metrics = evaluator.get_best_model()
    print(f"\nBest Model: {best_model}")
    print(f"Best Metrics: {best_metrics}")
    
    print("\nPipeline completed successfully!")

if __name__ == '__main__':
    main()
"""
src/models/train_all.py
批量训练脚本：对所有 (store, family) 序列跑 Seasonal Naive 基线 + 滚动验证。

用法：
    python -m src.models.train_all
    python -m src.models.train_all --n-sequences 20   # 只跑前20个序列（快速测试）
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

from src.models.utils import (
    load_time_series,
    time_split,
    evaluate_all,
    log_experiment,
    load_sequence_list,
)
from src.models.baseline import SeasonalNaive

warnings.filterwarnings("ignore")


def run_baseline_all(
    n_sequences: int = None,
    output_csv: str = "output/baseline_results.csv",
):
    """
    对所有序列跑 Seasonal Naive 基线，输出汇总指标表。
    """
    pairs = load_sequence_list()
    if n_sequences:
        pairs = pairs[:n_sequences]

    print(f"Running Seasonal Naive on {len(pairs)} sequences...")
    results = []
    t0 = time.time()

    for idx, (store, family) in enumerate(pairs):
        try:
            s = load_time_series(store, family, cols=["sales"])
            train, val, test = time_split(s)

            model = SeasonalNaive(period=7)
            model.fit(train["sales"])
            preds = model.predict(len(val))
            metrics = evaluate_all(val["sales"].values, preds)

            results.append({
                "store_nbr": store,
                "family": family,
                "rmsle": metrics["RMSLE"],
                "rmse": metrics["RMSE"],
                "mae": metrics["MAE"],
                "train_days": len(train),
                "val_days": len(val),
            })

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(pairs)}] RMSLE avg={np.mean([r['rmsle'] for r in results[-100:]]):.4f}")
        except Exception as e:
            print(f"  [{idx+1}/{len(pairs)}] SKIP ({store}, {family}): {e}")

    elapsed = time.time() - t0
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # 汇总
    summary = {
        "n_sequences": len(df),
        "rmsle_mean": df["rmsle"].mean(),
        "rmsle_median": df["rmsle"].median(),
        "rmsle_std": df["rmsle"].std(),
        "rmse_mean": df["rmse"].mean(),
        "mae_mean": df["mae"].mean(),
    }
    print(f"\n=== Baseline Summary ({len(df)} sequences) ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  runtime: {elapsed:.1f}s")
    print(f"  output: {output_csv}")

    # 记录实验日志
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
    store_nbr: int = 1,
    family: str = "GROCERY I",
    n_splits: int = 3,
    output_csv: str = None,
):
    """
    对单个序列跑滚动原点验证。
    """
    from src.models.utils import rolling_origin_split

    s = load_time_series(store_nbr, family, cols=["sales"])
    results = []
    t0 = time.time()

    for train, val, split_idx in rolling_origin_split(s, n_splits=n_splits):
        model = SeasonalNaive(period=7)
        model.fit(train["sales"])
        preds = model.predict(len(val))
        metrics = evaluate_all(val["sales"].values, preds)
        results.append({
            "split": split_idx,
            "train_end": train.index[-1].strftime("%Y-%m-%d"),
            "val_start": val.index[0].strftime("%Y-%m-%d"),
            "val_end": val.index[-1].strftime("%Y-%m-%d"),
            **metrics,
        })
        print(f"  Split {split_idx}: val={val.index[0].strftime('%Y-%m-%d')}~{val.index[-1].strftime('%Y-%m-%d')}  RMSLE={metrics['RMSLE']:.4f}")

    elapsed = time.time() - t0
    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)

    log_experiment(
        experiment_name="rolling_validation",
        model_name="SeasonalNaive",
        params={"period": 7, "n_splits": n_splits,
                "store_nbr": store_nbr, "family": family},
        metrics={"rmsle_mean": df["RMSLE"].mean()},
        runtime_seconds=elapsed,
        extra={"details": output_csv},
    )
    print(f"  Rolling avg RMSLE: {df['RMSLE'].mean():.4f}  (runtime: {elapsed:.1f}s)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Train all sequences with Seasonal Naive baseline")
    parser.add_argument("--n-sequences", type=int, default=None,
                        help="Number of sequences to process (default: all)")
    parser.add_argument("--rolling", action="store_true",
                        help="Run rolling validation on a representative sequence")
    parser.add_argument("--store", type=int, default=1,
                        help="Store for rolling validation")
    parser.add_argument("--family", type=str, default="GROCERY I",
                        help="Family for rolling validation")
    args = parser.parse_args()

    if args.rolling:
        run_rolling_validation(store_nbr=args.store, family=args.family)
    else:
        run_baseline_all(n_sequences=args.n_sequences)


if __name__ == "__main__":
    main()

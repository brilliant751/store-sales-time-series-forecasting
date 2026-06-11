"""
src/models/generate_submission.py
生成 test set 预测提交文件。
高效实现：数据加载一次，批量处理所有序列。

用法：
    python -m src.models.generate_submission
    python -m src.models.generate_submission --n-sequences 100
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

from src.models.utils import log_experiment, load_sequence_list

warnings.filterwarnings("ignore")


def generate_submission(
    n_sequences: int = None,
    output_path: str = "output/predictions.csv",
):
    """
    对所有序列使用 Seasonal Naive 预测并生成提交文件。
    一次性加载数据，避免每序列重复 I/O。
    """
    t0 = time.time()

    # 1. 一次性加载全部训练数据
    print("Loading feature dataset...")
    train = pd.read_csv("data/processed/feature_dataset.csv",
                         usecols=["date", "store_nbr", "family", "sales",
                                  "dataset_split"])
    train = train[train["dataset_split"] == "train"].copy()
    train["date"] = pd.to_datetime(train["date"])
    train = train.sort_values(["store_nbr", "family", "date"])
    print(f"  Loaded {len(train)} training rows")

    # 2. 加载 test 框架
    test_template = pd.read_csv("data/test.csv")
    print(f"  Loaded {len(test_template)} test rows")

    # 3. 逐个序列预测
    pairs = load_sequence_list()
    if n_sequences:
        pairs = pairs[:n_sequences]

    predictions = []
    for idx, (store, family) in enumerate(pairs):
        try:
            series = train[(train["store_nbr"] == store)
                          & (train["family"] == family)]["sales"].values

            if len(series) < 7:
                # 数据不足，用全局均值
                pred_val = float(train["sales"].mean())
                mask = ((test_template["store_nbr"] == store)
                       & (test_template["family"] == family))
                ids = test_template.loc[mask, "id"].values
                for pid in ids:
                    predictions.append({"id": int(pid), "sales": round(pred_val, 2)})
                continue

            # Seasonal Naive: 用最后 7 天重复
            last_7 = series[-7:]
            repeats = (16 + 6) // 7
            preds = np.tile(last_7, repeats)[:16]

            mask = ((test_template["store_nbr"] == store)
                   & (test_template["family"] == family))
            ids = test_template.loc[mask, "id"].values
            for i, pid in enumerate(ids):
                predictions.append({"id": int(pid), "sales": round(float(preds[i]), 2)})

            if (idx + 1) % 200 == 0:
                print(f"  [{idx+1}/{len(pairs)}]")
        except Exception as e:
            print(f"  SKIP ({store}, {family}): {e}")

    elapsed = time.time() - t0
    sub = pd.DataFrame(predictions).sort_values("id").reset_index(drop=True)
    sub.to_csv(output_path, index=False)

    log_experiment(
        experiment_name="submission_baseline",
        model_name="SeasonalNaive",
        params={"period": 7},
        metrics={},
        runtime_seconds=elapsed,
        extra={"n_sequences": len(pairs), "output": output_path},
    )

    print(f"\n✅ Saved {len(sub)} predictions to {output_path}")
    print(f"   Runtime: {elapsed:.1f}s")
    return sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sequences", type=int, default=None)
    parser.add_argument("--output", default="output/predictions.csv")
    args = parser.parse_args()
    generate_submission(args.n_sequences, args.output)


if __name__ == "__main__":
    main()

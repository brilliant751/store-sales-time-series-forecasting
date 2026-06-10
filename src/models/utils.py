"""
src/models/utils.py
工具函数：评估指标、时间切分、数据加载
"""
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Optional

DATA_DIR = "data"
PROCESSED_DIR = f"{DATA_DIR}/processed"


def load_time_series(
    store_nbr: int,
    family: str,
    cols: List[str] = None,
    data_dir: str = PROCESSED_DIR,
) -> pd.DataFrame:
    """
    从 feature_dataset.csv 加载单个 (store, family) 时序序列。

    Parameters
    ----------
    store_nbr : int
        门店编号
    family : str
        商品家族名称
    cols : list, optional
        要选择的列（默认: ['sales', 'onpromotion', 'oil_price',
                          'transactions', 'is_holiday', 'is_event']）
    data_dir : str
        处理后数据目录

    返回 DataFrame，date 设为 index。
    仅返回 dataset_split='train' 的记录（含真实 sales）。
    """
    if cols is None:
        cols = ["sales", "onpromotion", "oil_price",
                "transactions", "is_holiday", "is_event"]

    df = pd.read_csv(f"{data_dir}/feature_dataset.csv")
    mask = (
        (df["store_nbr"] == store_nbr)
        & (df["family"] == family)
        & (df["dataset_split"] == "train")
    )
    series = df[mask][["date"] + cols].copy()
    series["date"] = pd.to_datetime(series["date"])
    series = series.sort_values("date").set_index("date")
    return series


def load_oil(data_dir: str = PROCESSED_DIR) -> pd.Series:
    """
    从 feature_dataset 提取油价序列。
    返回 date 为索引的 Series。
    """
    df = pd.read_csv(f"{data_dir}/feature_dataset.csv",
                     usecols=["date", "oil_price"])
    oil = df.drop_duplicates("date").set_index("date").squeeze()
    oil.index = pd.to_datetime(oil.index)
    oil = oil.sort_index()
    return oil


def time_split(
    df: pd.DataFrame,
    val_start: str = "2017-07-01",
    test_start: str = "2017-08-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按日期切分训练/验证/测试集。
    返回 (train, val, test)。
    """
    train = df[df.index < val_start]
    val = df[(df.index >= val_start) & (df.index < test_start)]
    test = df[df.index >= test_start]
    return train, val, test


def rolling_origin_split(
    df: pd.DataFrame,
    n_splits: int = 3,
    val_days: int = 31,
    min_train_days: int = 365,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
    """
    滚动原点验证切分。
    每次训练窗口前移一个验证期长度，生成 (train, val, split_idx)。

    Parameters
    ----------
    df : pd.DataFrame
        date 为 index 的时序数据
    n_splits : int
        滚动轮数
    val_days : int
        每轮验证天数
    min_train_days : int
        最少训练天数（第1轮至少留这么多数据）
    """
    dates = df.index.sort_values()
    max_date = dates[-1]
    for i in range(n_splits):
        val_end = max_date - pd.Timedelta(days=(n_splits - 1 - i) * val_days)
        val_start = val_end - pd.Timedelta(days=val_days)
        train_end = val_start
        train_start = dates[0]
        # 过滤训练数据不足的情况
        n_train = len(df[train_start:train_end])
        if n_train < min_train_days:
            continue
        yield df[train_start:train_end], df[val_start:val_end], i


def log_experiment(
    experiment_name: str,
    model_name: str,
    params: dict,
    metrics: Dict[str, float],
    runtime_seconds: float,
    extra: dict = None,
    log_dir: str = "reports/experiments",
) -> str:
    """
    记录实验日志到 JSON 文件。

    Parameters
    ----------
    experiment_name : str
        实验名称（如 "baseline_all_sequences"）
    model_name : str
        模型名称（如 "SeasonalNaive"）
    params : dict
        模型参数
    metrics : dict
        评估指标 (RMSLE, RMSE, MAE)
    runtime_seconds : float
        运行时长（秒）
    log_dir : str
        日志保存目录

    Returns
    -------
    str: 日志文件路径
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    record = {
        "experiment": experiment_name,
        "model": model_name,
        "params": params,
        "metrics": metrics,
        "runtime_seconds": round(runtime_seconds, 2),
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        record["extra"] = extra

    filename = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = log_path / filename
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return str(filepath)


def load_sequence_list(data_dir: str = PROCESSED_DIR, split: str = "train") -> List[Tuple[int, str]]:
    """从 feature_dataset 加载所有 (store_nbr, family) 对"""
    df = pd.read_csv(f"{data_dir}/feature_dataset.csv",
                     usecols=["store_nbr", "family", "dataset_split"])
    df = df[df["dataset_split"] == split]
    pairs = df[["store_nbr", "family"]].drop_duplicates().values.tolist()
    return [(int(s), str(f)) for s, f in pairs]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根对数误差（主指标）"""
    return float(
        np.sqrt(np.mean((np.log1p(y_true) - np.log1p(np.maximum(y_pred, 0))) ** 2))
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差"""
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_all(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """返回包含三个指标的 dict"""
    return {
        "RMSLE": rmsle(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }


def build_holiday_df(
    holidays_path: str = f"{DATA_DIR}/holidays_events.csv",
) -> pd.DataFrame:
    """
    将 holidays_events.csv 转换为 Prophet 要求的 holidays DataFrame。

    处理规则见设计文档：
    - Work Day: 跳过（补班日不算假日）
    - Transfer: 跳过（原始日期不算）
    - Holiday/Additional/Bridge/Event: 保留
    - locale == 'Local' 设 lower_scale=0.5
    """
    holidays = pd.read_csv(holidays_path)
    prophet_holidays = []

    for _, row in holidays.iterrows():
        if row["type"] in ("Work Day",):
            continue
        if row["transferred"]:
            continue

        h = {
            "holiday": row["description"],
            "ds": pd.to_datetime(row["date"]),
            "lower_window": 0,
            "upper_window": 0,
        }
        if row["locale"] == "Local":
            h["scale"] = 0.5
        prophet_holidays.append(h)

    return pd.DataFrame(prophet_holidays)

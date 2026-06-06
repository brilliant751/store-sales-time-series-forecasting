"""
src/models/utils.py
工具函数：评估指标、时间切分、数据加载
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

DATA_DIR = "data"


def load_time_series(
    store_nbr: int, family: str, data_dir: str = DATA_DIR
) -> pd.DataFrame:
    """
    从 train.csv 加载单个 (store, family) 时序序列。

    返回 DataFrame，列: ['sales', 'onpromotion']
    date 解析为 datetime 并设为 index。
    """
    train = pd.read_csv(f"{data_dir}/train.csv")
    mask = (train["store_nbr"] == store_nbr) & (train["family"] == family)
    series = train[mask][["date", "sales", "onpromotion"]].copy()
    series["date"] = pd.to_datetime(series["date"])
    series = series.sort_values("date").set_index("date")
    return series


def load_oil(data_dir: str = DATA_DIR) -> pd.Series:
    """
    加载油价数据，返回 date 为索引的 Series。
    缺失值向前填充。
    """
    oil = pd.read_csv(f"{data_dir}/oil.csv")
    oil["date"] = pd.to_datetime(oil["date"])
    oil = oil.set_index("date").squeeze()
    oil = oil.sort_index()
    oil = oil.ffill()
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
            "ds": row["date"],
            "lower_window": 0,
            "upper_window": 0,
        }
        if row["locale"] == "Local":
            h["scale"] = 0.5
        prophet_holidays.append(h)

    return pd.DataFrame(prophet_holidays)

"""
src/models/baseline.py
Seasonal Naive 基线模型
"""
import numpy as np
import pandas as pd


class SeasonalNaive:
    """
    季节性朴素预测：用上一个周期同时段的值作为预测。

    Parameters
    ----------
    period : int
        周期长度（默认7天，周季节性）
    """

    def __init__(self, period: int = 7):
        self.period = period
        self._last_season: pd.Series = None

    def fit(self, y: pd.Series):
        """保存最后一个完整周期的值"""
        if len(y) < self.period:
            raise ValueError(
                f"Series length ({len(y)}) < period ({self.period})"
            )
        self._last_season = y.iloc[-self.period:].copy()
        return self

    def predict(self, steps: int) -> np.ndarray:
        """
        预测未来 steps 步。
        重复最后一个周期的模式。
        """
        if self._last_season is None:
            raise RuntimeError("Model not fitted yet")
        repeats = (steps + self.period - 1) // self.period
        full = np.tile(self._last_season.values, repeats)
        return full[:steps]

"""
src/models/prophet_model.py
Prophet 模型封装（含节假日特征）
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet


class ProphetModel:
    """
    Prophet 模型封装。

    自动集成：
    - 厄瓜多尔节假日表
    - onpromotion 作为额外回归量
    - 油价 (dcoilwtico) 作为额外回归量
    """

    def __init__(
        self,
        holidays_df: pd.DataFrame = None,
        extra_regressors: List[str] = None,
        yearly_seasonality: str = "auto",
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
    ):
        self.holidays_df = holidays_df
        self.extra_regressors = extra_regressors or []
        self._model = None

        self._model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
        )

        if holidays_df is not None and len(holidays_df) > 0:
            self._model.holidays = holidays_df

    def fit(self, df: pd.DataFrame):
        """
        训练 Prophet 模型。

        Parameters
        ----------
        df : pd.DataFrame
            必须包含列: 'ds' (日期), 'y' (目标值)
            可选包含 extra_regressors 指定的列
        """
        for reg in self.extra_regressors:
            self._model.add_regressor(reg)

        # 确保无 NaN，避免 Prophet 报错
        df = df.dropna().reset_index(drop=True)
        self._model.fit(df)
        return self

    def predict(self, periods: int, future_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        预测未来 periods 步。

        Parameters
        ----------
        periods : int
            预测步数
        future_df : pd.DataFrame, optional
            包含未来日期和 extra regressor 值的 DataFrame

        Returns
        -------
        pd.DataFrame: 包含 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
        """
        if future_df is not None:
            future = future_df.copy()
            # 对 extra regressors 中的 NaN 做前向填充（油价值日数据周末缺失）
            for reg in self.extra_regressors:
                if reg in future.columns:
                    future[reg] = future[reg].ffill().bfill()
        else:
            future = self._model.make_future_dataframe(periods=periods)

        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def get_yhat(self, forecast: pd.DataFrame) -> np.ndarray:
        """从 forecast DataFrame 提取 yhat 并截断到非负"""
        return np.maximum(forecast["yhat"].values, 0)

    def prepare_data(
        self,
        series: pd.DataFrame,
        oil_series: pd.Series = None,
    ) -> pd.DataFrame:
        """
        将我们的时序数据转为 Prophet 所需的格式。

        Parameters
        ----------
        series : pd.DataFrame
            包含 'sales' 和 'onpromotion' 列，date 为 index
        oil_series : pd.Series, optional
            油价数据

        Returns
        -------
        pd.DataFrame: 包含 'ds', 'y' 和 extra_regressors
        """
        df = series.reset_index()[["date", "sales", "onpromotion"]].rename(
            columns={"date": "ds", "sales": "y"}
        )
        if oil_series is not None and "dcoilwtico" in self.extra_regressors:
            oil_df = oil_series.reset_index()
            oil_df.columns = ["ds", "dcoilwtico"]
            oil_df["ds"] = pd.to_datetime(oil_df["ds"])
            df = df.merge(oil_df, on="ds", how="left")
            df["dcoilwtico"] = df["dcoilwtico"].ffill()
        return df

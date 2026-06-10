"""
src/models/sarima.py
SARIMA 模型封装
"""
import itertools
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class SarimaModel:
    """
    SARIMA 模型封装。

    Parameters
    ----------
    order : tuple
        (p, d, q)
    seasonal_order : tuple
        (P, D, Q, s)
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self._model = None
        self._result = None

    def fit(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        disp: bool = False,
    ):
        """训练 SARIMA 模型"""
        self._model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = self._model.fit(disp=disp, maxiter=200)
        return self

    def predict(
        self,
        steps: int,
        exog: Optional[pd.DataFrame] = None,
        dynamic: bool = False,
    ) -> np.ndarray:
        """预测未来 steps 步"""
        if self._result is None:
            raise RuntimeError("Model not fitted yet")
        forecast = self._result.forecast(steps=steps, exog=exog)
        return np.maximum(forecast.values, 0)

    def summary(self) -> str:
        """返回模型摘要"""
        if self._result is None:
            return "Model not fitted"
        return str(self._result.summary())

    @staticmethod
    def grid_search(
        y: pd.Series,
        p_range: List[int] = None,
        d_range: List[int] = None,
        q_range: List[int] = None,
        P_range: List[int] = None,
        D_range: List[int] = None,
        Q_range: List[int] = None,
        s: int = 7,
        exog: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict, float]:
        """
        网格搜索最优 SARIMA 参数（AIC 选优）。

        返回 (best_params, best_aic)
        """
        p_range = p_range or [0, 1, 2]
        d_range = d_range or [0, 1]
        q_range = q_range or [0, 1, 2]
        P_range = P_range or [0, 1]
        D_range = D_range or [0, 1]
        Q_range = Q_range or [0, 1]

        best_aic = float("inf")
        best_params = None

        for p, d, q, P, D, Q in itertools.product(
            p_range, d_range, q_range, P_range, D_range, Q_range
        ):
            try:
                model = SARIMAX(
                    y,
                    exog=exog,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=200)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_params = {
                        "order": (p, d, q),
                        "seasonal_order": (P, D, Q, s),
                    }
            except Exception:
                continue

        if best_params is None:
            raise RuntimeError("Grid search failed: no valid model found")

        return best_params, best_aic

# 成员2 前置准备实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为薛漕洋搭建 ARIMA/SARIMA + Prophet 统计模型代码框架，并跑通单序列建模流程

**Architecture:** 5 个 Python 模块 (`src/models/`) + 2 个 Jupyter notebook (`notebooks/`)，模块间通过 `utils.py` 共享评估指标与数据加载逻辑，每个模型类有统一的 `fit/predict` 接口

**Tech Stack:** Python 3.8+, pandas, numpy, scikit-learn, statsmodels, prophet, matplotlib

---

### Task 0: 目录结构与依赖安装

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/` (目录)
- Create: `notebooks/` (目录)
- Create: `output/` (目录)

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
mkdir -p src/models notebooks output
```

- [ ] **Step 2: 创建 `__init__.py`**

```python
# src/models/__init__.py
# 统计模型模块：SARIMA / Prophet / Seasonal Naive
```

- [ ] **Step 3: 更新 requirements.txt**

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
prophet>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
pytest>=6.0.0
```

- [ ] **Step 4: 安装依赖**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
pip install -r requirements.txt
```

- [ ] **Step 5: Commit**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
git add src/models/__init__.py requirements.txt
git commit -m "chore: init models dir structure and deps

Co-Authored-By: AtomCode (deepseek-v4-flash) <noreply@atomgit.com>"
```

---

### Task 1: `utils.py` — 评估指标、时间切分、数据加载

**Files:**
- Create: `src/models/utils.py`

本模块提供所有模型共用的工具函数。三个评估指标按照竞赛标准实现。

- [ ] **Step 1: 编写 `utils.py`**

```python
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

    返回 DataFrame，列: ['date', 'sales', 'onpromotion']
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
    oil = oil.fillna(method="ffill")
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
```

- [ ] **Step 2: 快速验证**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
python3 -c "
from src.models.utils import (
    load_time_series, time_split, evaluate_all, rmsle, rmse, mae, build_holiday_df
)
import numpy as np

s = load_time_series(1, 'GROCERY I')
print(f'Series shape: {s.shape}')
print(f'Date range: {s.index.min()} ~ {s.index.max()}')

train, val, test = time_split(s)
print(f'Train: {train.shape}, Val: {val.shape}, Test: {test.shape}')

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 2.2, 2.8])
metrics = evaluate_all(y_true, y_pred)
print(f'Metrics: {metrics}')

hol = build_holiday_df()
print(f'Holiday rows: {len(hol)}')
print(f'Holiday columns: {list(hol.columns)}')
"
```

Expected output:
```
Series shape: (1688, 2)
Date range: 2013-01-01 ~ 2017-08-15
Train: (1642, 2), Val: (31, 2), Test: (15, 2)
Metrics: {'RMSLE': ..., 'RMSE': ..., 'MAE': ...}
Holiday rows: 300+ (depends on filtering)
Holiday columns: ['holiday', 'ds', 'lower_window', 'upper_window']
```

- [ ] **Step 3: Commit**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
git add src/models/utils.py
git commit -m "feat: add utils with metrics, time split, data loading, holiday builder

Co-Authored-By: AtomCode (deepseek-v4-flash) <noreply@atomgit.com>"
```

---

### Task 2: `baseline.py` — Seasonal Naive 基线

**Files:**
- Create: `src/models/baseline.py`

Seasonal Naive：用前一周同一天的值作为本周该天的预测值。

- [ ] **Step 1: 编写 `baseline.py`**

```python
"""
src/models/baseline.py
Seasonal Naive 基线模型
"""
import pandas as pd
import numpy as np


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
```

- [ ] **Step 2: 验证**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
python3 -c "
from src.models.utils import load_time_series, time_split, evaluate_all
from src.models.baseline import SeasonalNaive

s = load_time_series(1, 'GROCERY I')
train, val, test = time_split(s)

model = SeasonalNaive(period=7)
model.fit(train['sales'])
preds = model.predict(len(val))
metrics = evaluate_all(val['sales'].values, preds)
print(f'Seasonal Naive Val Metrics: {metrics}')
"
```

Expected output:
```
Seasonal Naive Val Metrics: {'RMSLE': ..., 'RMSE': ..., 'MAE': ...}
```

- [ ] **Step 3: Commit**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
git add src/models/baseline.py
git commit -m "feat: add Seasonal Naive baseline

Co-Authored-By: AtomCode (deepseek-v4-flash) <noreply@atomgit.com>"
```

---

### Task 3: `sarima.py` — SARIMA 模型封装

**Files:**
- Create: `src/models/sarima.py`

使用 statsmodels SARIMAX，封装 fit/predict/grid_search。

- [ ] **Step 1: 编写 `sarima.py`**

```python
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
```

- [ ] **Step 2: 验证（快速测试，只跑少量参数）**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
python3 -c "
from src.models.utils import load_time_series, time_split, evaluate_all
from src.models.sarima import SarimaModel

s = load_time_series(1, 'GROCERY I')
# 只用 2017 年数据做快速验证（~200 天）
train = s.loc['2017-01-01':'2017-06-30']
val = s.loc['2017-07-01':'2017-07-31']

model = SarimaModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
model.fit(train['sales'])
preds = model.predict(steps=len(val))
metrics = evaluate_all(val['sales'].values, preds)
print(f'SARIMA(1,1,1)(1,1,1,7) Val Metrics: {metrics}')

# 快速网格搜索（极小范围）
params, aic = SarimaModel.grid_search(
    train['sales'],
    p_range=[0, 1], d_range=[0, 1], q_range=[0, 1],
    P_range=[0, 1], D_range=[0, 1], Q_range=[0, 1],
    s=7,
)
print(f'Best params: {params}, AIC: {aic:.2f}')
"
```

Expected output:
```
SARIMA(1,1,1)(1,1,1,7) Val Metrics: {'RMSLE': ..., 'RMSE': ..., 'MAE': ...}
Best params: {...}, AIC: ...
```

- [ ] **Step 3: Commit**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
git add src/models/sarima.py
git commit -m "feat: add SARIMA model with grid search

Co-Authored-By: AtomCode (deepseek-v4-flash) <noreply@atomgit.com>"
```

---

### Task 4: `prophet_model.py` — Prophet 模型封装

**Files:**
- Create: `src/models/prophet_model.py`

封装 Prophet，集成厄瓜多尔节假日和 extra regressors。

- [ ] **Step 1: 编写 `prophet_model.py`**

```python
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
            df["dcoilwtico"] = df["dcoilwtico"].fillna(method="ffill")
        return df
```

- [ ] **Step 2: 验证**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
python3 -c "
from src.models.utils import load_time_series, time_split, evaluate_all, build_holiday_df, load_oil
from src.models.prophet_model import ProphetModel

s = load_time_series(1, 'GROCERY I')
train, val, test = time_split(s)
oil = load_oil()

holidays_df = build_holiday_df()

model = ProphetModel(
    holidays_df=holidays_df,
    extra_regressors=['onpromotion', 'dcoilwtico'],
)

train_df = model.prepare_data(train, oil_series=oil)
model.fit(train_df)

future = val.reset_index()[['date', 'onpromotion']].rename(columns={'date': 'ds'})
# 合并油价
oil_df = oil.reset_index()
oil_df.columns = ['ds', 'dcoilwtico']
future = future.merge(oil_df, on='ds', how='left')
future['dcoilwtico'] = future['dcoilwtico'].fillna(method='ffill')

forecast = model.predict(periods=0, future_df=future)
y_pred = model.get_yhat(forecast)
metrics = evaluate_all(val['sales'].values, y_pred)
print(f'Prophet Val Metrics: {metrics}')
"
```

Expected output:
```
Prophet Val Metrics: {'RMSLE': ..., 'RMSE': ..., 'MAE': ...}
```

- [ ] **Step 3: Commit**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
git add src/models/prophet_model.py
git commit -m "feat: add Prophet model with holiday and regressor support

Co-Authored-By: AtomCode (deepseek-v4-flash) <noreply@atomgit.com>"
```

---

### Task 5: 原型 Notebook — SARIMA 单序列跑通

**Files:**
- Create: `notebooks/02a_prototype-sarima.ipynb`

Notebook cell-by-cell 内容（不输出 base64 大文件）：

**Cell 1 — 标题与导入:**
```markdown
# SARIMA 单序列原型

**目的**：对代表性序列跑通 SARIMA 建模流程，验证时间切分、评估接口、网格搜索。

**序列**：
1. (1, GROCERY I) — 主力高销量
2. (1, AUTOMOTIVE) — 稀疏序列
```

**Cell 2 — 导入:**
```python
import sys, os, warnings
sys.path.append(os.path.abspath('..'))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.models.utils import load_time_series, time_split, evaluate_all
from src.models.baseline import SeasonalNaive
from src.models.sarima import SarimaModel

%matplotlib inline
plt.style.use('ggplot')
```

**Cell 3 — 加载并可视化 S1:**
```python
s1 = load_time_series(1, 'GROCERY I')
train, val, test = time_split(s1)

fig, ax = plt.subplots(figsize=(14, 4))
train['sales'].plot(ax=ax, label='Train')
val['sales'].plot(ax=ax, label='Validation')
test['sales'].plot(ax=ax, label='Test')
ax.set_title('Store 1, GROCERY I — Sales')
ax.legend()
plt.tight_layout()
plt.savefig('../output/sarima_s1_timeseries.png', dpi=100)
plt.show()

print(f'Train: {train.index.min()} ~ {train.index.max()}  ({len(train)} days)')
print(f'Val:   {val.index.min()} ~ {val.index.max()}    ({len(val)} days)')
print(f'Test:  {test.index.min()} ~ {test.index.max()}  ({len(test)} days)')
```

**Cell 4 — Seasonal Naive 基线:**
```python
baseline = SeasonalNaive(period=7)
baseline.fit(train['sales'])
preds_seas = baseline.predict(len(val))
metrics_seas = evaluate_all(val['sales'].values, preds_seas)
print('Seasonal Naive Val:', metrics_seas)
```

**Cell 5 — 网格搜索最佳 SARIMA 参数:**
```python
# 用 2016-01 后的数据加速搜索
train_sub = train.loc['2016-01-01':]
print(f'Grid search on {len(train_sub)} days...')

best_params, best_aic = SarimaModel.grid_search(
    train_sub['sales'],
    p_range=[0,1,2], d_range=[0,1], q_range=[0,1,2],
    P_range=[0,1], D_range=[0,1], Q_range=[0,1],
    s=7,
)
print(f'Best params: {best_params}')
print(f'Best AIC: {best_aic:.2f}')
```

**Cell 6 — 训练最佳 SARIMA 并验证:**
```python
model = SarimaModel(**best_params)
model.fit(train['sales'])
preds = model.predict(steps=len(val))
metrics_val = evaluate_all(val['sales'].values, preds)
print('SARIMA Val:', metrics_val)

# 对比
comparison = pd.DataFrame({
    'Seasonal Naive': metrics_seas,
    'SARIMA': metrics_val,
})
print('\nComparison:\n', comparison)
```

**Cell 7 — 可视化预测 vs 真实值:**
```python
fig, ax = plt.subplots(figsize=(14, 5))
val['sales'].plot(ax=ax, label='Actual', linewidth=2)
pd.Series(preds, index=val.index).plot(ax=ax, label='SARIMA Pred', linestyle='--')
pd.Series(preds_seas, index=val.index).plot(ax=ax, label='Seasonal Naive', linestyle=':')
ax.set_title('SARIMA — Validation Predictions vs Actual')
ax.legend()
plt.tight_layout()
plt.savefig('../output/sarima_s1_prediction.png', dpi=100)
plt.show()
```

**Cell 8 — 测试集预测:**
```python
model.fit(train['sales'])  # refit on all train
preds_test = model.predict(steps=len(test))
metrics_test = evaluate_all(test['sales'].values, preds_test)
print('SARIMA Test:', metrics_test)
```

**Cell 9 — S2 (AUTOMOTIVE) 快速验证:**
```python
s2 = load_time_series(1, 'AUTOMOTIVE')
t2, v2, te2 = time_split(s2)

params2, aic2 = SarimaModel.grid_search(
    t2.loc['2016-01-01':]['sales'],
    p_range=[0,1], d_range=[0,1], q_range=[0,1],
    P_range=[0,1], D_range=[0,1], Q_range=[0,1],
    s=7,
)
print(f'S2 best params: {params2}, AIC: {aic2:.2f}')

m2 = SarimaModel(**params2)
m2.fit(t2['sales'])
p2 = m2.predict(steps=len(v2))
m2_val = evaluate_all(v2['sales'].values, p2)
print(f'S2 SARIMA Val: {m2_val}')
```

- [ ] **Step 1: 创建 notebook 文件**

直接用 Jupyter 的 `.ipynb` 格式写入。实际内容会用 JSON，每个 cell 一个 dict。

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
touch notebooks/02a_prototype-sarima.ipynb
```

鉴于 notebook 文件较大，写一个 Python 脚本来生成它会更可控。或者直接创建一个最小可用的 .ipynb JSON 文件。

- [ ] **Step 2: 运行 notebook 验证**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
jupyter nbconvert --to script notebooks/02a_prototype-sarima.ipynb --stdout | python3
```

- [ ] **Step 3: Commit**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
git add notebooks/02a_prototype-sarima.ipynb
git commit -m "feat: add SARIMA prototype notebook

Co-Authored-By: AtomCode (deepseek-v4-flash) <noreply@atomgit.com>"
```

---

### Task 6: 原型 Notebook — Prophet 单序列跑通

**Files:**
- Create: `notebooks/02b_prototype-prophet.ipynb`

结构与 SARIMA notebook 类似。

- [ ] **Step 1: 创建 notebook 文件**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
touch notebooks/02b_prototype-prophet.ipynb
```

- [ ] **Step 2: Commit**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
git add notebooks/02b_prototype-prophet.ipynb
git commit -m "feat: add Prophet prototype notebook (skeleton)

Co-Authored-By: AtomCode (deepseek-v4-flash) <noreply@atomgit.com>"
```

---

### Task 7: 整体验证

- [ ] **Step 1: 运行全量验证脚本**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
python3 -c "
from src.models.utils import load_time_series, time_split, evaluate_all
from src.models.baseline import SeasonalNaive
from src.models.sarima import SarimaModel
from src.models.prophet_model import ProphetModel

print('All modules imported successfully')

# 测试 3 个代表性序列的 Seasonal Naive
for store, family in [(1, 'GROCERY I'), (1, 'AUTOMOTIVE'), (44, 'BEVERAGES')]:
    s = load_time_series(store, family)
    train, val, test = time_split(s)
    model = SeasonalNaive(period=7)
    model.fit(train['sales'])
    preds = model.predict(len(val))
    metrics = evaluate_all(val['sales'].values, preds)
    print(f'SNaive ({store}, {family}): RMSLE={metrics[\"RMSLE\"]:.4f}')
"
```

Expected output: 正常打印三个序列的指标值。

- [ ] **Step 2: 最终 commit**

```bash
cd /Users/brilliant751/Desktop/数据分析与挖掘/final/project
git add -A
git commit -m "feat: complete stat model framework with validation

Co-Authored-By: AtomCode (deepseek-v4-flash) <noreply@atomgit.com>"
```

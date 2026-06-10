# 成员2 前置准备设计文档

> 作者：薛漕洋（2350240）
> 创建日期：2025-06-06
> 对应角色：算法负责人（统计模型方向）

## 一、背景与目标

本项目为 Kaggle Store Sales - Time Series Forecasting 竞赛的课程实现。成员2负责 ARIMA/SARIMA、Prophet 统计模型的建模与调优。本文档覆盖数据就绪后、成员1数据清洗完成前的**前置准备阶段**设计。

### 前置准备范围

1. 阅读 Kaggle 数据说明，明确字段含义
2. 搭建 ARIMA/SARIMA 和 Prophet 的代码框架
3. 用原始 train.csv 跑通单序列建模流程
4. 确定时间切分方式与评估接口
5. 设计 Prophet 节假日特征接入与 SARIMA 季节周期假设

## 二、数据理解

### 2.1 数据文件概览

| 文件 | 行数 | 大小 | 作用 |
|------|------|------|------|
| `train.csv` | 3,000,888 | 116 MB | 主训练集 (2013-01-01 ~ 2017-08-15) |
| `test.csv` | 28,512 | 1.0 MB | 待预测日期 (2017-08-16 ~ 2017-08-31, 16天) |
| `stores.csv` | 54 | 1.4 KB | 门店元数据 (5 types, 17 clusters) |
| `oil.csv` | 1,218 | 20 KB | 每日油价 (43个缺失值, 3.5%) |
| `holidays_events.csv` | 350 | 22 KB | 节假日与事件 (6种类型) |
| `transactions.csv` | 83,488 | 1.5 MB | 每日每店总交易量 |
| `sample_submission.csv` | 28,512 | 334 KB | 提交格式模板 |

### 2.2 关键数据特征

- **维度**：54 stores × 33 families = 1782 个时序序列
- **稀疏性**：31.3% 的 sales=0，大量（store, family）组合在特定日期无销售
- **油价缺失**：43/1218 天缺失，需前向填充
- **节假日类型**：Holiday / Transfer / Additional / Bridge / Work Day / Event
- **地震事件**：2016-04-16 厄瓜多尔 7.8 级地震，后续几周销量异常
- **工资发放**：公共部门工资每月 15 日和最后一天发放，影响超市销量

## 三、代码框架

### 3.1 目录结构

```
src/models/
├── __init__.py           # 模块初始化
├── baseline.py           # Seasonal Naive 基线
├── sarima.py             # SARIMA 模型封装
├── prophet_model.py      # Prophet 模型封装（含节假日）
└── utils.py              # 评估指标、时间切分、数据加载工具

notebooks/
├── 02a_prototype-sarima.ipynb    # 单序列 SARIMA 原型
└── 02b_prototype-prophet.ipynb   # 单序列 Prophet 原型
```

### 3.2 模块职责

**`utils.py`**:
- `load_time_series(store_nbr, family)`：从 train.csv 加载单序列
- `time_split(df, val_start, test_start)`：按日期切分训练/验证/测试
- `rmsle(y_true, y_pred)` / `rmse(y_true, y_pred)` / `mae(y_true, y_pred)`：三个指标
- `evaluate_all(y_true, y_pred)`：返回包含 RMSLE/RMSE/MAE 的 dict

**`baseline.py`**:
- `SeasonalNaive(period=7)`：用前一周期同时段的值作为预测，作为最低可比基线

**`sarima.py`**:
- `SarimaModel(order, seasonal_order)`：封装 statsmodels SARIMAX
- `fit(train_data, exog=None)`：训练
- `predict(steps, exog=None)`：预测未来 steps 步
- `grid_search(train_data, p_range, d_range, q_range, P_range, D_range, Q_range, s)`：AIC 网格搜索选参

**`prophet_model.py`**:
- `ProphetModel(holidays_df, extra_regressors)`：封装 Prophet
- `build_holiday_df(holidays_events_df)`：转换节假日表
- `fit(train_df)`：训练（含节假日 + extra regressors）
- `predict(future_df)`：预测

## 四、时间切分策略

### 4.1 固定时间窗口

```
训练集: 2013-01-01 ~ 2017-06-30   (约 4.5 年)
验证集: 2017-07-01 ~ 2017-07-31   (1 个月 holdout)
测试集: 2017-08-01 ~ 2017-08-15   (模拟竞赛 15 天窗口)
```

### 4.2 滚动验证（扩展）

在开发阶段选 1 个代表性序列跑 3 次滚动验证，每次训练窗口前移 1 个月：

| 轮次 | 训练截止 | 验证 |
|------|----------|------|
| 1 | 2017-05-31 | 2017-06 |
| 2 | 2017-06-30 | 2017-07 |
| 3 | 2017-07-31 | 2017-08 |

## 五、代表性序列选择

开发阶段不跑全部 1782 个序列，选择 3 个代表性序列：

| 编号 | (store, family) | 特征 |
|------|-----------------|------|
| S1 | (1, GROCERY I) | 高销量、强周季节性的主力序列 |
| S2 | (1, AUTOMOTIVE) | 极低销量、零值多，测试模型鲁棒性 |
| S3 | (44, BEVERAGES) | 稀疏序列但有促销事件 |

## 六、SARIMA 季节周期假设

### 6.1 周期设定

| 周期 | 参数 s | 依据 |
|------|--------|------|
| 周周期（默认） | s=7 | 零售数据星期几模式明显 |
| 年周期（扩展） | s=365 | 需要降采样到周数据或近似处理 |

### 6.2 网格搜索范围

```
p ∈ {0,1,2}, d ∈ {0,1}, q ∈ {0,1,2}
P ∈ {0,1}, D ∈ {0,1}, Q ∈ {0,1}
s = 7（周周期，默认）
```

选优标准：AIC（小范围网格搜索避免组合爆炸）

## 七、Prophet 节假日特征接入

### 7.1 节假日表构建规则

| holidays_events 类型 | 处理方式 |
|----------------------|----------|
| Holiday | 正常加入 holidays 表 |
| Transfer | **跳过**（原始日期不算假，实际庆祝日在对应 Transfer 行） |
| Additional | 视为普通假日加入 |
| Bridge | 视为普通假日加入 |
| Work Day | **跳过**（补班日，不算假日） |
| Event | 视为区域性事件，加入 holidays 表 |

### 7.2 区域权重

- `locale == 'National'`：标准权重
- `locale == 'Regional'`：降低权重 (`scale` 可调)
- `locale == 'Local'`：进一步降低权重

### 7.3 Extra Regressors

- `onpromotion`：促销商品数量，作为 Prophet 额外回归量
- `dcoilwtico`：油价，先向前填充缺失值，再作为回归量

## 八、交付物

1. 本设计文档 (`docs/superpowers/specs/2025-06-06-member2-prep-design.md`)
2. `src/models/` 下的统计模型代码框架（模块化、可独立运行）
3. `notebooks/` 下的 2 个原型 notebook
4. 跑通单序列 SARIMA 和 Prophet 建模流程的验证结果

## 九、与成员1的依赖关系

- **成员1（数据清洗）** 产出：`data/processed/` 中的清洗后数据
- **成员2（本阶段）** 使用原始 `data/train.csv` 开发原型，后续切换至成员1的处理后数据
- **接口约定**：只要确保 `store_nbr`, `family`, `date`, `sales`, `onpromotion` 这些核心字段名不变，数据流水线替换不影响模型代码

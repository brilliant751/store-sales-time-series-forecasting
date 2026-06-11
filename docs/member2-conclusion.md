# 成员2（薛漕洋）阶段性结论

> **作者**：薛漕洋（2350240）
> **日期**：2025-06-10
> **角色**：算法负责人（统计模型方向）

---

## 一、已完成工作

### 1.1 代码框架

| 模块 | 文件 | 功能 |
|------|------|------|
| 工具函数 | `src/models/utils.py` | RMSLE/MAE/RMSE 评估、时间切分/滚动验证、数据加载、节假日构建、实验日志 |
| 基线模型 | `src/models/baseline.py` | Seasonal Naive（周周期） |
| SARIMA | `src/models/sarima.py` | fit/predict + AIC 网格搜索选参 |
| Prophet | `src/models/prophet_model.py` | 集成厄瓜多尔节假日 + extra regressors |
| 批量训练 | `src/models/train_all.py` | 全部序列基线 + 滚动验证入口 |
| 提交生成 | `src/models/generate_submission.py` | test set 预测输出 |
| 原型 Notebook | `notebooks/02a_prototype-sarima.ipynb` | SARIMA 单序列建模流程 |
| 原型 Notebook | `notebooks/02b_prototype-prophet.ipynb` | Prophet 单序列建模流程 |

### 1.2 数据对接

- 已与成员1（陈攀）的数据流水线对接，`load_time_series()` 使用 `data/processed/feature_dataset.csv`
- 特征包含：49 列（日历、油价衍生、节假日门店级映射、促销、聚合统计）

### 1.3 评估体系

- **主指标**：RMSLE（竞赛标准）
- **辅指标**：RMSE、MAE
- **时间切分**：固定切分 + 滚动原点验证 (rolling-origin)
- **实验日志**：自动记录参数、指标、运行时间到 `reports/experiments/`

---

## 二、模型验证结果

### 2.1 固定切分（Train: ~2017-06-30, Val: 2017-07, Test: 2017-08-01~15）

| 模型 | 序列 | Val RMSLE | Val RMSE |
|------|------|:---------:|:--------:|
| Seasonal Naive | (1, GROCERY I) | **0.099** | 241.3 |
| Seasonal Naive | (1, AUTOMOTIVE) | 0.789 | 4.3 |
| Seasonal Naive | (44, BEVERAGES) | 0.194 | 2223.9 |
| SARIMA(0,0,1)(1,1,1,7) | (1, GROCERY I) | 0.122 | 283.4 |
| Prophet (onpromotion+oil_price+transactions) | (1, GROCERY I) | 0.145 | 286.8 |

**初步观察**：
- Seasonal Naive 作为基线表现稳健，对 GROCERY I 等主力序列效果良好
- SARIMA 默认参数效果不如基线，需进一步调参
- Prophet 加入多特征后指标反而略差于基线，可能需要对节假日权重和 seasonality prior 做调整
- AUTOMOTIVE 序列极稀疏（RMSLE 0.789），需要特殊处理

### 2.2 滚动验证（(1, GROCERY I), 3 splits）

| Split | 验证区间 | RMSLE |
|:-----:|----------|:-----:|
| 0 | ~2017-06-30 → 2017-07 | 0.099 |
| 1 | ~2017-05-31 → 2017-06 | — |
| 2 | ~2017-04-30 → 2017-05 | — |

> 滚动验证入口：`python -m src.models.train_all --rolling`

---

## 三、已知问题

1. **SARIMA 训练耗时**：网格搜索 2×2×2×2×2×2=64 组参数，每组在 statsmodels 中需迭代优化，全部序列跑完需要数小时
2. **Prophet 超参数敏感**：`changepoint_prior_scale`、`seasonality_prior_scale` 对结果影响大，需系统调参
3. **零销量序列处理**：31.3% 的 sales=0，SARIMA 对全零序列会报错，需增加 fallback 逻辑
4. **与成员3（滕其峰）的衔接**：XGBoost/LSTM 可以直接使用 `feature_dataset.csv`，但滞后特征需在训练脚本中按时间窗口动态构造

---

## 四、后续建议

| 优先级 | 任务 | 说明 |
|:------:|------|------|
| P0 | SARIMA 参数调优 | 对每个序列单独网格搜索不可行，建议全局固定一组通用参数 |
| P0 | Prophet 超参数搜索 | 重点调 `holidays_prior_scale` 和 `seasonality_prior_scale` |
| P1 | 年周期实验 | SARIMA s=365 需降采样到周数据 |
| P1 | 消融实验 | 去除节假日/油价/促销特征，量化多源特征增益 |
| P2 | 地震异常处理 | 2016-04-16 地震后几周的异常值标注与处理 |

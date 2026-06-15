# Store Sales Time Series Forecasting

这个项目的主流程是：

`data/` 原始 CSV -> `data_pipeline` -> `data/processed/feature_dataset.csv` -> `src/models/train_all.py`

## 数据

我已经确认项目可以直接读取 `data/train.csv`，也就是原始数据目录是可用的。

## 环境

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行

先跑数据管道：

```bash
python -m data_pipeline.cli
```

再跑训练入口。默认训练三个模型并比较：

```bash
python -m src.models.train_all
```

````

## 训练入口

`src/models/train_all.py` 现在会：

1. 运行 `data_pipeline`
2. 生成 `data/processed/feature_dataset.csv`
3. 训练 `XGBoost`、`LSTM`、`GRU`
4. 生成对比图和对比报告

可选参数：

```bash
python -m src.models.train_all --model baseline
python -m src.models.train_all --model xgboost
python -m src.models.train_all --model lstm
python -m src.models.train_all --model gru
python -m src.models.train_all --model all
python -m src.models.train_all --skip-pipeline --model xgboost
python -m src.models.train_all --skip-pipeline --model xgboost --n-sequences 20
python -m src.models.train_all --dl-sequences 10
python -m src.models.train_all --rolling
````

说明：LSTM/GRU 默认只抽样少量 `store/family` 序列训练，避免把整张特征表展开成超大 3D 张量。

## 输出

- `data/processed/`：数据管道产物
- `output/baseline_results.csv`：基线结果汇总
- `output/xgboost_metrics.csv`：XGBoost 训练与验证指标
- `output/lstm_metrics.csv`：LSTM 训练与验证指标
- `output/gru_metrics.csv`：GRU 训练与验证指标
- `outputs/models/xgboost_model.pkl`：XGBoost 模型
- `outputs/models/xgboost_feature_importance.csv`：XGBoost 特征重要性
- `outputs/models/lstm_model.h5`：LSTM 模型
- `outputs/models/gru_model.h5`：GRU 模型
- `outputs/figures/model_comparison.png`：三模型对比图
- `outputs/model_comparison_report.md`：三模型对比报告
- `reports/experiments/`：实验日志

## 训练与可视化

训练模型时会自动生成可视化图表（默认启用）：

```bash
# 训练所有模型并生成可视化（默认行为）
python -m src.models.train_all

# 训练并跳过可视化
python -m src.models.train_all --no-visualize

# 单独生成可视化（使用真实数据）
python -m src.evaluation.visualization

# 可视化支持的参数
python -m src.evaluation.visualization --store 1 --family "GROCERY I"
python -m src.evaluation.visualization --start-date 2017-01-01 --end-date 2017-06-30
```

### 可视化图表

#### 静态图表（保存到 `output/figures/`）

| 图表文件 | 说明 |
|----------|------|
| `sales_history.png` | 销售历史折线图 |
| `model_comparison.png` | XGBoost、LSTM、GRU模型性能对比柱状图 |
| `metrics_radar.png` | 模型性能雷达图 |
| `store_sales_bar.png` | 商店销售额柱状图 |
| `family_sales_bar.png` | 商品类别销售额柱状图 |
| `store_family_heatmap.png` | 商店-商品类别热力图 |
| `metrics_heatmap.png` | 模型指标热力图 |
| `sales_boxplot.png` | 销售分布箱线图 |
| `prediction_scatter.png` | 预测值与真实值散点图 |
| `residual_distribution.png` | 残差分布图 |
| `predicted_sales_timeseries.png` | 预测销售量时间序列图 |
| `predicted_sales_distribution.png` | 预测销售量分布图 |
| `predicted_sales_by_family.png` | 各商品类别预测销售量对比图 |

#### 交互式图表（保存到 `output/interactive/`）

| 图表文件 | 说明 |
|----------|------|
| `sales_history_interactive.html` | 交互式销售历史图（支持时间范围选择） |
| `model_comparison_interactive.html` | 交互式模型对比图（支持指标切换） |
| `heatmap_interactive.html` | 交互式热力图（支持数据筛选） |

### 评估指标

- **RMSLE**（均方根对数误差）- 主要评估指标
- **MAE**（平均绝对误差）
- **RMSE**（均方根误差）
- **MAPE**（平均绝对百分比误差）
- **R²**（决定系数）

### 模型对比

可视化脚本会自动对比以下三个模型：
- **XGBoost** - 梯度提升树模型
- **LSTM** - 长短期记忆网络
- **GRU** - 门控循环单元

评估报告自动保存到 `output/evaluation_report.md`

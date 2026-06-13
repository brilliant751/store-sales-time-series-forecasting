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

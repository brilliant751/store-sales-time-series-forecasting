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

再跑训练入口：

```bash
python -m src.models.train_all
```

一条命令跑完整流程：

```bash
make setup
make train-all
```

## 训练入口

`src/models/train_all.py` 现在会：

1. 运行 `data_pipeline`
2. 生成 `data/processed/feature_dataset.csv`
3. 基于该特征集训练 Seasonal Naive 基线并输出结果

可选参数：

```bash
python -m src.models.train_all --n-sequences 20
python -m src.models.train_all --rolling
python -m src.models.train_all --skip-pipeline
```

## 输出

- `data/processed/`：数据管道产物
- `output/baseline_results.csv`：基线结果汇总
- `reports/experiments/`：实验日志


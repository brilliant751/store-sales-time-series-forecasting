# Store Sales Time Series Forecasting

这是一个基于 Kaggle `Store Sales - Time Series Forecasting` 数据集的销售预测项目，当前代码包含：

- XGBoost 训练与预测
- LSTM / GRU 深度学习训练与预测
- 模型指标对比与报告生成

## 数据检查

我已经确认项目可以读取 `data/train.csv`。

`train.csv` 当前可正常被 `pandas.read_csv()` 读取，前几列为：

- `id`
- `date`
- `store_nbr`
- `family`
- `sales`
- `onpromotion`

当前代码主要直接使用：

- `data/train.csv`
- `data/test.csv`

其余文件如 `stores.csv`、`oil.csv`、`holidays_events.csv`、`transactions.csv` 已放在 `data/` 下，但目前训练主流程里还没有全部接入。

## 环境准备

建议使用 Python 3.10+。

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux 激活：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果你在 Windows 上没有把 `python` 加入 PATH，也可以直接用 `py -3`。

## Makefile

项目根目录已提供 `Makefile`，常用命令如下：

```bash
make setup
make train-all
make predict-all
make all
```

说明：

- `make setup`：创建 `.venv` 并安装依赖
- `make train-all`：训练 XGBoost、LSTM、GRU，并生成评估报告
- `make predict-all`：使用已训练模型生成预测结果
- `make all`：依次执行训练和预测

单独运行某个模型也可以：

```bash
make train-xgboost
make train-lstm
make train-gru
make predict-xgboost
make predict-lstm
make predict-gru
```

## 手动运行

如果你不想用 `make`，也可以直接运行：

```bash
python src/models/train_all.py
python src/models/xgboost/train.py
python src/models/xgboost/predict.py
python src/models/deep_learning/train.py
python src/models/deep_learning/predict.py
```

## 输出结果

运行后会生成：

- `outputs/models/`：模型文件
- `outputs/predictions/`：预测结果
- `outputs/figures/`：对比图
- `logs/`：训练日志


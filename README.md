# 基于多源时序数据的门店-商品销量预测

同济大学 商务智能案例分析 课程期末项目

## 项目简介

本项目旨在基于多源时序数据（销售历史、促销/活动、门店信息、节假日与外部天气等）对门店-商品组合的短期销量进行预测，支持补货与促销决策。实现了数据处理、特征工程、基线模型训练与评估流程，并提供可复现的实验步骤。

## 仓库结构

- `data/`：原始与处理后数据（示例/说明，实际数据视课程要求放置）。
- `notebooks/`：探索性分析与建模笔记本。
- `src/`：项目代码（数据处理、特征、模型训练与预测脚本）。
- `configs/`：配置文件（如模型参数、路径）。
- `output/`：训练产生的模型与结果输出（预测文件、可视化）。
- `README.md`：项目说明（本文件）。

> 注：若某些目录为空，请根据课程提交规范将数据与脚本放置相应位置。

## 方法概览

1. 数据清洗与对齐：处理缺失、对齐多源时序（按日/周/月聚合）。
2. 特征工程：历史滞后特征、滑动窗口统计、节假日/促销/门店/商品类目等外部特征。
3. 模型：基线模型（如随机森林、XGBoost）与时序模型（如LSTM/Transformer-based）供对比。
4. 评估：使用MAE、RMSE等指标对验证集进行评估，并按门店/商品细分汇报结果。

## 运行说明

建议使用 Python 虚拟环境（`venv` 或 `conda`），示例命令：

```bash
# 创建虚拟环境（venv）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖（若提供 requirements.txt）
pip install -r requirements.txt
```

常见执行流程（根据仓库实际脚本调整）：

```bash
# 数据预处理
python src/preprocess.py --input data/raw --output data/processed

# 特征工程
python src/feature_engineering.py --input data/processed --output data/features

# 训练并评估模型
python src/train.py --config configs/train_config.yaml

# 生成预测
python src/predict.py --model output/model.pkl --output output/predictions.csv
```

## 结果与提交

- 结果文件位于 `output/`，包含预测表与评估报告（示例：`predictions.csv`, `evaluation.csv`）。
- 请确保将最终提交的预测文件与简要报告按课程要求打包提交。

## 依赖与环境

- 推荐 Python 3.8+
- 常用库：`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`, `torch`（如使用深度学习）

## 注意事项

- 数据隐私：若使用真实门店数据，请在提交前移除或脱敏敏感信息。
- 运行长时间训练任务时请在有 GPU 的环境下执行并调整 batch/epoch 等参数。

## 联系与贡献

如需修改或改进，请在仓库内提交 issue 或 pull request，或联系项目负责人（课程提交时填写）。

---

如果你希望我根据仓库内已有脚本自动生成更精确的 `运行说明` 与命令，我可以继续扫描 `src/` 与 `notebooks/` 并把 `README.md` 进一步细化。

# 第三部分：机器学习/深度学习模型实现报告

## 负责人

滕其峰（2354159）

## 完成内容

### 1. 项目结构创建

创建了完整的项目目录结构：

```
store-sales-time-series-forecasting/
├── src/
│   ├── models/
│   │   ├── xgboost/
│   │   │   ├── model.py        # XGBoost模型类定义
│   │   │   ├── train.py        # XGBoost训练脚本
│   │   │   └── predict.py      # XGBoost预测脚本
│   │   ├── deep_learning/
│   │   │   ├── model.py        # LSTM/GRU模型类定义
│   │   │   ├── train.py        # 深度学习训练脚本
│   │   │   └── predict.py      # 深度学习预测脚本
│   │   └── train_all.py        # 统一训练入口
│   └── evaluation/
│       └── evaluator.py        # 评估与可视化模块
├── outputs/
│   ├── models/                 # 保存训练好的模型
│   ├── predictions/            # 保存预测结果
│   └── figures/                # 保存可视化图表
├── logs/                       # 训练日志
└── requirements.txt            # 依赖清单
```

### 2. XGBoost模型实现

**核心功能：**

- 基于XGBoost的回归模型
- 支持时间序列交叉验证
- 特征重要性分析（使用SHAP值）
- 超参数调优接口
- 多指标评估（RMSLE、MAE、RMSE）

**特征工程：**

- 滞后特征：lag(1, 7, 14, 28)
- 滚动统计：均值、标准差（7天、14天窗口）
- 时间特征：星期、月份、季度、是否周末
- 类别特征：门店、商品类别（独热编码）

### 3. 深度学习模型实现

**LSTM/GRU模型：**

- 支持LSTM和GRU两种网络结构
- 多层神经网络设计（默认2层）
- Dropout正则化防止过拟合
- Early Stopping早停机制
- 序列长度可配置（默认28天）

**网络架构：**

```
Input (sequence_length, n_features)
↓
LSTM Layer 1 (128 units, return_sequences=True)
↓
Dropout (0.2)
↓
LSTM Layer 2 (64 units)
↓
Dropout (0.2)
↓
Dense Layer (32 units, ReLU)
↓
Output Layer (1 unit)
```

### 4. 评估与可视化模块

**评估功能：**

- 多指标计算（RMSLE、MAE、RMSE）
- 模型对比分析
- 残差分析
- 特征重要性可视化

**输出图表：**

- 预测值vs真实值对比图
- 残差分布图
- 特征重要性条形图
- 模型性能对比图

## 项目启动方式

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 如果需要深度学习支持
pip install tensorflow keras
```

### 数据准备

将Kaggle竞赛数据放置在 `data/` 目录下：

- `train.csv` - 训练数据
- `test.csv` - 测试数据

### 训练模型

**方式一：训练所有模型**

```bash
python src/models/train_all.py
```

**方式二：单独训练XGBoost**

```bash
python -m src.models.xgboost.train
```

**方式三：单独训练LSTM**

```bash
python -m src.models.deep_learning.train --model_type lstm
```

**方式四：单独训练GRU**

```bash
python -m src.models.deep_learning.train --model_type gru
```

### 生成预测

**XGBoost预测：**

```bash
python -m src.models.xgboost.predict
```

**LSTM预测：**

```bash
python -m src.models.deep_learning.predict --model_type lstm
```

**GRU预测：**

```bash
python -m src.models.deep_learning.predict --model_type gru
```

### 输出文件

训练完成后，生成以下输出：

```
outputs/
├── models/
│   ├── xgboost_model.pkl      # XGBoost模型
│   ├── lstm_model.h5          # LSTM模型
│   └── gru_model.h5           # GRU模型
├── predictions/
│   ├── xgboost_predictions.csv
│   ├── lstm_predictions.csv
│   └── gru_predictions.csv
├── figures/
│   └── model_comparison.png   # 模型对比图
└── model_comparison_report.md # 对比报告
```

## 关键配置参数

### XGBoost参数

```python
{
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 10,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

### 深度学习参数

```python
{
    'units': 128,           # LSTM/GRU单元数
    'layers': 2,            # 网络层数
    'dropout_rate': 0.2,    # Dropout比例
    'learning_rate': 0.001, # 学习率
    'sequence_length': 28,  # 序列长度
    'epochs': 50,           # 训练轮数
    'batch_size': 64        # 批次大小
}
```

## 评估指标说明

| 指标  | 说明           | 计算公式                                    |
| ----- | -------------- | ------------------------------------------- |
| RMSLE | 均方根对数误差 | sqrt(mean((log(y_true+1)-log(y_pred+1))^2)) |
| MAE   | 平均绝对误差   | mean(\|y_true - y_pred\|)                   |
| RMSE  | 均方根误差     | sqrt(mean((y_true - y_pred)^2))             |

## 代码特点

1. **模块化设计**：模型、训练、预测分离，便于维护和扩展
2. **可配置性**：关键参数可通过配置文件或命令行参数调整
3. **错误处理**：完善的异常处理和日志记录
4. **可复现性**：固定随机种子，确保实验可重复
5. **扩展性**：支持添加新模型和特征工程方法

## 后续优化建议

1. **超参数调优**：使用Optuna进行贝叶斯优化
2. **特征增强**：添加更多外生特征（油价、节假日等）
3. **模型集成**：尝试Stacking或Blending集成方法
4. **分布式训练**：使用Dask或Spark处理大规模数据
5. **模型解释**：增加SHAP/LIME可解释性分析

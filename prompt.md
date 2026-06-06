# 机器学习/深度学习模型构建计划

## 任务概述

**负责人**: 滕其峰(2354159)
**角色**: 算法负责人（机器学习/深度学习方向）
**核心任务**: XGBoost、LSTM/GRU 建模、特征选择与训练优化

## 项目结构规划

```
store-sales-time-series-forecasting/
├── data/
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后的数据
│   └── features/               # 特征工程输出
├── src/
│   ├── data_pipeline/          # 数据预处理（与薛漕洋协作）
│   ├── models/
│   │   ├── xgboost/            # XGBoost 模型
│   │   │   ├── __init__.py
│   │   │   ├── model.py        # 模型定义
│   │   │   ├── train.py        # 训练脚本
│   │   │   ├── predict.py      # 预测脚本
│   │   │   └── tune.py         # 超参数调优
│   │   ├── deep_learning/      # 深度学习模型
│   │   │   ├── __init__.py
│   │   │   ├── lstm_model.py   # LSTM 模型
│   │   │   ├── gru_model.py    # GRU 模型
│   │   │   ├── train.py        # 训练脚本
│   │   │   ├── predict.py      # 预测脚本
│   │   │   └── tune.py         # 超参数调优
│   │   └── utils.py            # 模型工具函数
│   ├── evaluation/             # 评估模块（与滕其峰协作）
│   └── config/                 # 配置文件
│       └── model_config.yaml   # 模型配置
├── notebooks/
│   ├── 01_xgboost_experiments.ipynb
│   ├── 02_lstm_experiments.ipynb
│   ├── 03_feature_selection.ipynb
│   └── 04_model_comparison.ipynb
├── logs/
│   ├── xgboost_tuning.log      # XGBoost 调参日志
│   ├── lstm_tuning.log         # LSTM 调参日志
│   └── training_history/       # 训练历史
├── outputs/
│   ├── models/                 # 保存的模型
│   ├── predictions/            # 预测结果
│   └── figures/                # 对比分析图表
└── requirements.txt
```

## 第一阶段：环境准备与数据理解（第1-2周）

### 1.1 环境配置

- [ ] 安装依赖包：xgboost, tensorflow/pytorch, scikit-learn, pandas, numpy
- [ ] 配置 GPU 环境（如使用深度学习）
- [ ] 设置随机种子确保可复现性

### 1.2 数据探索

- [ ] 分析数据分布特征
- [ ] 检查缺失值和异常值
- [ ] 理解时序数据的周期性模式
- [ ] 与数据工程负责人协作，确保特征工程满足模型需求

## 第二阶段：XGBoost 模型开发（第2-3周）

### 2.1 基础模型构建

- [ ] 实现 XGBoost 回归模型
- [ ] 设计特征输入格式
- [ ] 实现时间序列交叉验证
- [ ] 建立 baseline 模型性能

### 2.2 特征工程

- [ ] 滞后特征：lag(1, 7, 14, 28)
- [ ] 滚动统计特征：滚动均值、标准差、最大值、最小值
- [ ] 时间特征：星期、月份、季度、是否工作日
- [ ] 外生特征：油价、促销、节假日
- [ ] 门店特征：城市、类型、聚类

### 2.3 特征选择

- [ ] 使用特征重要性分析
- [ ] 递归特征消除（RFE）
- [ ] SHAP 值分析特征贡献
- [ ] 消融实验验证特征有效性

### 2.4 超参数调优

- [ ] 网格搜索/随机搜索
- [ ] 贝叶斯优化（Optuna）
- [ ] 交叉验证评估
- [ ] 记录调参日志

**关键调参参数**:

- n_estimators, max_depth, learning_rate
- subsample, colsample_bytree
- reg_alpha, reg_lambda
- min_child_weight

## 第三阶段：深度学习模型开发（第3-4周）

### 3.1 LSTM 模型

- [ ] 数据序列化处理
- [ ] 设计 LSTM 网络架构
- [ ] 实现多变量时间序列输入
- [ ] 添加 Dropout 防止过拟合
- [ ] 实现早停机制

**LSTM 架构建议**:

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

### 3.2 GRU 模型

- [ ] 实现 GRU 网络架构
- [ ] 对比 LSTM 与 GRU 性能
- [ ] 优化网络结构
- [ ] 训练过程监控

### 3.3 深度学习调优

- [ ] 网络层数与单元数优化
- [ ] 学习率调度
- [ ] Batch size 优化
- [ ] 激活函数选择
- [ ] 正则化参数调整

## 第四阶段：模型对比与优化（第4周）

### 4.1 统一评估

- [ ] 在相同测试集上评估所有模型
- [ ] 计算主要指标：RMSLE
- [ ] 计算辅助指标：MAE, RMSE
- [ ] 生成性能对比表格

### 4.2 可视化分析

- [ ] 预测值 vs 真实值对比图
- [ ] 残差分析图
- [ ] 特征重要性可视化
- [ ] 模型性能对比柱状图
- [ ] 时序预测效果图

### 4.3 误差分析

- [ ] 分析不同门店的预测误差
- [ ] 分析不同产品类别的预测误差
- [ ] 分析节假日和促销期间的误差
- [ ] 识别系统性偏差

### 4.4 模型集成（可选）

- [ ] 简单平均集成
- [ ] 加权平均集成
- [ ] Stacking 集成
- [ ] 对比集成效果

## 第五阶段：交付准备（第5周）

### 5.1 代码整理

- [ ] 代码模块化重构
- [ ] 添加详细注释
- [ ] 编写函数文档
- [ ] 确保代码可运行

### 5.2 输出文件准备

- [ ] 模型文件保存（.pkl, .h5 等）
- [ ] 预测结果文件
- [ ] 调参日志整理
- [ ] 可视化图表导出

### 5.3 文档编写

- [ ] 模型使用说明
- [ ] 关键发现总结
- [ ] 模型优缺点分析
- [ ] 改进建议

## 关键输出物清单

### 代码文件

1. `src/models/xgboost/model.py` - XGBoost 模型定义
2. `src/models/xgboost/train.py` - XGBoost 训练脚本
3. `src/models/xgboost/tune.py` - XGBoost 调参脚本
4. `src/models/deep_learning/lstm_model.py` - LSTM 模型
5. `src/models/deep_learning/gru_model.py` - GRU 模型
6. `src/models/deep_learning/train.py` - 深度学习训练脚本
7. `src/models/utils.py` - 工具函数

### 日志文件

1. `logs/xgboost_tuning.log` - XGBoost 调参详细记录
2. `logs/lstm_tuning.log` - LSTM 调参详细记录
3. `logs/training_history/` - 训练过程记录

### 分析图表

1. 模型性能对比图（RMSLE, MAE, RMSE）
2. 特征重要性图（XGBoost）
3. SHAP 值分析图
4. 预测效果时序图
5. 残差分析图
6. 训练过程损失曲线

### 实验记录

1. 各模型最佳参数配置
2. 消融实验结果
3. 运行时间统计
4. 模型对比分析报告

## 技术要点

### XGBoost 关键技术

- 时间序列交叉验证（TimeSeriesSplit）
- 处理多序列建模（门店-商品组合）
- 特征工程与特征选择
- 超参数优化策略

### 深度学习关键技术

- 序列数据预处理（归一化、序列化）
- 多变量时间序列建模
- 防止过拟合（Dropout, Early Stopping）
- 学习率调度策略

### 评估与对比

- 统一评估框架
- 多维度性能分析
- 统计显著性检验
- 可解释性分析

## 协作要点

### 与薛漕洋（数据工程）协作

- 确认特征工程输出格式
- 反馈特征需求
- 协调数据预处理流程

### 与朱从周（统计模型）协作

- 统一评估指标
- 对比不同方法性能
- 分享实验发现

### 与滕其峰（评估分析）协作

- 提供模型预测结果
- 协助可视化分析
- 整合对比分析报告

## 风险与应对

### 风险1：深度学习训练时间过长

- **应对**: 使用较小的网络结构作为基线，采用早停机制，考虑使用预训练模型

### 风险2：特征工程效果不佳

- **应对**: 与数据工程负责人紧密协作，尝试多种特征组合，使用自动化特征选择

### 风险3：模型过拟合

- **应对**: 严格的时间序列交叉验证，使用正则化，增加训练数据

### 风险4：GPU 资源不足

- **应对**: 优先使用 XGBoost，深度学习模型使用较小的网络结构

## 成功标准

1. ✅ 完成 XGBoost 和 LSTM/GRU 两种模型的完整实现
2. ✅ 模型性能达到或超过基线模型
3. ✅ 提供详细的调参日志和实验记录
4. ✅ 生成完整的模型对比分析图表
5. ✅ 代码可复现，文档清晰
6. ✅ 与团队成员协作顺畅，按时交付

## 时间节点

- **第2周结束**: 完成 XGBoost baseline 模型
- **第3周结束**: 完成 XGBoost 调优和深度学习基线模型
- **第4周结束**: 完成所有模型调优和对比分析
- **第5周中期**: 完成代码整理和文档编写
- **第5周结束**: 完成交付物准备

---

**注**: 本计划将根据项目进展和团队协作情况进行动态调整。

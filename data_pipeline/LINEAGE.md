# 数据血缘说明

## 总览

本数据血缘针对 `data_pipeline/` 生成的四类产物：

- `integrated_dataset.csv`
- `feature_dataset.csv`
- `quality_report.json`
- `lineage_metadata.json`

整体流程为：`原始 CSV -> 标准化清洗 -> 门店粒度事件映射 -> 多表融合 -> 特征派生 -> 质量与血缘导出`

## 源表与角色

### `train.csv`

- 粒度：`date x store_nbr x family`
- 作用：训练样本主表，提供 `sales` 与 `onpromotion`
- 关键处理：补齐完整时间网格，缺失销量与促销填 0

### `test.csv`

- 粒度：`date x store_nbr x family`
- 作用：待预测样本主表
- 关键处理：保留 `onpromotion`，并在融合层追加静态与外部特征

### `stores.csv`

- 粒度：`store_nbr`
- 作用：补充门店静态属性 `city/state/type/cluster`
- 关键处理：文本字段统一转小写

### `oil.csv`

- 粒度：`date`
- 作用：提供外部市场变量 `oil_price`
- 关键处理：补齐全日期、线性插值、前后向填补、生成移动平均和一阶差分

### `holidays_events.csv`

- 粒度：事件记录
- 作用：提供节假日/事件/补班信息
- 关键处理：
  - 删除 `transferred=True` 的记录
  - 按 `locale` 将节假日映射为国家级、州级、城市级门店特征
  - 聚合到 `date x store_nbr`

### `transactions.csv`

- 粒度：`date x store_nbr`
- 作用：提供门店交易量
- 关键处理：扩展到完整 `date x store_nbr` 粒度，缺失值填 0 并保留缺失标记

## 处理链路

### 1. 标准化清洗

- 统一日期字段为日期类型
- 统一 `store_nbr`、`cluster` 等字段类型
- 文本类字段做小写与空白规范化

### 2. 训练网格补齐

- 对 `train.csv` 构造完整 `date x store_nbr x family` 网格
- 新增样本以 `is_generated_row=1` 标记
- 将补齐出的 `sales`、`onpromotion` 填补为 0

### 3. 外部表补全与映射

- `oil.csv`：按全日期补全并插值
- `transactions.csv`：按全 `date x store_nbr` 粒度补全
- `holidays_events.csv`：映射到门店层并聚合

### 4. 多表融合

按如下顺序融合：

1. 训练补齐表与测试表拼接为统一样本框架
2. 关联 `stores.csv`
3. 关联油价特征
4. 关联交易量特征
5. 关联门店级节假日特征

最终得到 `integrated_dataset.csv`

### 5. 特征派生

在融合表基础上生成：

- 日历特征
- 促销激活特征
- 节假日计数/命中特征
- 门店与商品家族轻量聚合统计特征

最终得到 `feature_dataset.csv`

## 输出物依赖关系

- `integrated_dataset.csv`
  - 直接依赖：`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`
- `feature_dataset.csv`
  - 直接依赖：`integrated_dataset.csv`
- `quality_report.json`
  - 直接依赖：所有源表、清洗中间结果、最终输出表
- `lineage_metadata.json`
  - 直接依赖：管道配置与预定义血缘描述

## 可追溯字段

以下字段用于支持追踪与回溯：

- `id`：原始竞赛记录主键
- `dataset_split`：区分 train/test 来源
- `is_generated_row`：标记是否由时间网格补齐生成
- `transactions_missing_original`：原始交易缺失标记
- `oil_missing_original`：原始油价缺失标记

## 运行方式

```bash
python -m data_pipeline.cli
```

如需自定义路径：

```bash
python -m data_pipeline.cli --raw-dir "e:\Desktop\course project\store-sales-time-series-forecasting\store-sales-time-series-forecasting" --output-dir "e:\Desktop\course project\store-sales-time-series-forecasting\data_pipeline\outputs"
```

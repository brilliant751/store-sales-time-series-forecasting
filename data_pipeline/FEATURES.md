# 特征文档

## 目标

`data_pipeline/` 输出的 `feature_dataset.csv` 面向后续建模与分析，粒度为 `date x store_nbr x family x dataset_split`。  
其中 `dataset_split=train` 的记录带有真实 `sales`，`dataset_split=test` 作为待预测样本保留 `sales` 为空。

## 特征分组

### 1. 主键与样本标识

- `id`：原始比赛样本 ID，训练集中补齐出的缺失日期记录为空。
- `date`：业务日期。
- `dataset_split`：样本来源，取值为 `train` 或 `test`。
- `store_nbr`：门店编号。
- `family`：商品家族。
- `store_family_key`：`store_nbr` 与 `family` 拼接后的组合键。

### 2. 目标与促销特征

- `sales`：销量目标，仅训练样本有值。
- `onpromotion`：当天促销商品数。
- `promo_active`：是否存在促销，`onpromotion > 0` 时为 1。
- `is_generated_row`：训练集补齐完整时间网格时新增的样本标记。

### 3. 交易与外部市场特征

- `transactions`：门店日交易量，来自 `transactions.csv`。
- `transactions_missing_original`：原始交易表缺失后被填补为 0 的标记。
- `oil_price`：插值后的日油价，原始列 `dcoilwtico` 标准化后的名称。
- `oil_missing_original`：原始油价缺失标记。
- `oil_price_change_1`：油价一阶差分。
- `oil_price_ma_7`：7 日油价移动平均。
- `oil_price_ma_28`：28 日油价移动平均。

### 4. 门店静态属性

- `city`：门店城市，小写规范化。
- `state`：门店州/省，小写规范化。
- `type`：门店类型，小写规范化。
- `cluster`：门店聚类编号。

### 5. 节假日与事件特征

- `holiday_count`：当天对该门店生效的节假日/事件数量。
- `holiday_name_count`：当天生效的节假日名称数。
- `holiday_names`：生效节假日/事件描述，使用 `|` 拼接。
- `is_national_holiday`：是否命中国家级节假日/事件。
- `is_regional_holiday`：是否命中州级节假日/事件。
- `is_local_holiday`：是否命中城市级节假日/事件。
- `is_holiday`：是否存在类型为 `holiday` 的记录。
- `is_event`：是否存在类型为 `event` 的记录。
- `is_additional`：是否存在类型为 `additional` 的记录。
- `is_bridge`：是否存在类型为 `bridge` 的记录。
- `is_work_day`：是否存在类型为 `work day` 的记录。

### 6. 日历特征

- `year`：年份。
- `quarter`：季度。
- `month`：月份。
- `day`：日。
- `day_of_week`：星期索引，周一为 0。
- `day_of_year`：年内第几天。
- `week_of_year`：ISO 周序号。
- `is_weekend`：是否周末。
- `is_month_start`：是否月初。
- `is_month_end`：是否月末。
- `is_quarter_start`：是否季初。
- `is_quarter_end`：是否季末。
- `days_since_start`：自数据起始日以来的天数。

### 7. 轻量聚合统计特征

- `family_mean_onpromotion`：同商品家族平均促销强度。
- `store_mean_onpromotion`：同门店平均促销强度。
- `family_mean_transactions`：同商品家族对应样本的平均交易量。
- `store_mean_transactions`：同门店平均交易量。

## 设计说明

- 借鉴参考 notebook 的有效思路：完整时间网格补齐、油价插值、节假日按国家/州/城市映射到门店层。
- 当前默认不生成基于 `sales` 的滞后窗口特征，原因是这类特征在训练/推理阶段需要严格区分可用历史窗口，建议在建模脚本里按验证切分策略动态构造，以避免时间泄漏。
- 如果后续模型确定为树模型或全局序列模型，可在此基础上继续扩展 lag、rolling、expanding 等历史统计特征。

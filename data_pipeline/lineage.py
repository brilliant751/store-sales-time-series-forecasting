from __future__ import annotations

from pathlib import Path
from typing import Any


def build_lineage_metadata(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "sources": [
            {
                "name": "train",
                "path": str(raw_dir / "train.csv"),
                "grain": "date x store_nbr x family",
                "primary_keys": ["id"],
                "business_keys": ["date", "store_nbr", "family"],
            },
            {
                "name": "test",
                "path": str(raw_dir / "test.csv"),
                "grain": "date x store_nbr x family",
                "primary_keys": ["id"],
                "business_keys": ["date", "store_nbr", "family"],
            },
            {
                "name": "stores",
                "path": str(raw_dir / "stores.csv"),
                "grain": "store_nbr",
                "primary_keys": ["store_nbr"],
                "business_keys": ["store_nbr"],
            },
            {
                "name": "oil",
                "path": str(raw_dir / "oil.csv"),
                "grain": "date",
                "primary_keys": ["date"],
                "business_keys": ["date"],
            },
            {
                "name": "holidays_events",
                "path": str(raw_dir / "holidays_events.csv"),
                "grain": "holiday/event record",
                "primary_keys": [],
                "business_keys": ["date", "type", "locale", "locale_name", "description"],
            },
            {
                "name": "transactions",
                "path": str(raw_dir / "transactions.csv"),
                "grain": "date x store_nbr",
                "primary_keys": [],
                "business_keys": ["date", "store_nbr"],
            },
        ],
        "transformations": [
            "标准化日期与字段类型，统一文本字段为小写规范格式。",
            "将训练集补齐为完整的 date x store_nbr x family 网格，缺失销量/促销按 0 填补，并保留 is_generated_row 标记。",
            "将油价扩展到完整日期范围，执行线性插值与前后向填补，并生成变动与移动平均特征。",
            "剔除 transferred=True 的节假日记录，并按 national/regional/local 规则映射到门店粒度。",
            "将 transactions 扩展到完整 date x store_nbr 粒度，缺失值填 0，并保留缺失标记。",
            "融合 stores、oil、transactions、holiday features，生成 integrated_dataset。",
            "在 integrated_dataset 基础上派生日历、促销、门店与聚合统计特征，输出 feature_dataset。",
        ],
        "outputs": [
            {
                "name": "integrated_dataset",
                "path": str(output_dir / "integrated_dataset.csv"),
                "grain": "date x store_nbr x family x split",
            },
            {
                "name": "feature_dataset",
                "path": str(output_dir / "feature_dataset.csv"),
                "grain": "date x store_nbr x family x split",
            },
            {
                "name": "quality_report",
                "path": str(output_dir / "quality_report.json"),
                "grain": "pipeline run",
            },
            {
                "name": "lineage_metadata",
                "path": str(output_dir / "lineage_metadata.json"),
                "grain": "pipeline run",
            },
        ],
    }

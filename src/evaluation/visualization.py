"""
综合评估与可视化脚本 - 重构版
使用真实数据进行评估分析和可视化展示
支持交互式操作和多样化图表类型
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error, r2_score

# 尝试导入交互式可视化库
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False
    print("Warning: plotly not installed. Interactive visualizations will be skipped.")

warnings.filterwarnings("ignore")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 项目路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
INTERACTIVE_DIR = OUTPUT_DIR / "interactive"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)


class SalesDataLoader:
    """销售数据加载器 - 加载真实数据"""
    
    def __init__(self, data_dir=None, output_dir=None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        
    def load_train_data(self, nrows=None):
        """加载训练数据（历史销售数据）"""
        train_path = self.data_dir / "train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path, nrows=nrows)
            df['date'] = pd.to_datetime(df['date'])
            print(f"Loaded train data: {len(df)} rows")
            return df
        return None
    
    def load_test_data(self):
        """加载测试数据"""
        test_path = self.data_dir / "test.csv"
        if test_path.exists():
            df = pd.read_csv(test_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"Loaded test data: {len(df)} rows")
            return df
        return None
    
    def load_predictions(self):
        """加载模型预测结果"""
        pred_path = self.output_dir / "predictions.csv"
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            print(f"Loaded predictions: {len(df)} rows")
            return df
        return None
    
    def load_stores(self):
        """加载商店信息"""
        stores_path = self.data_dir / "stores.csv"
        if stores_path.exists():
            df = pd.read_csv(stores_path)
            print(f"Loaded stores data: {len(df)} stores")
            return df
        return None
    
    def load_model_metrics(self):
        """加载各模型评估指标"""
        metrics = {}
        
        # Baseline 模型指标
        baseline_path = self.output_dir / "baseline_results.csv"
        if baseline_path.exists():
            metrics['Baseline'] = pd.read_csv(baseline_path)
            print(f"Loaded Baseline metrics: {len(metrics['Baseline'])} records")
        
        # XGBoost 模型指标
        xgb_path = self.output_dir / "xgboost_metrics.csv"
        if xgb_path.exists():
            metrics['XGBoost'] = pd.read_csv(xgb_path)
            print(f"Loaded XGBoost metrics")
        
        # LSTM 模型指标
        lstm_path = self.output_dir / "lstm_metrics.csv"
        if lstm_path.exists():
            metrics['LSTM'] = pd.read_csv(lstm_path)
            print(f"Loaded LSTM metrics")
        
        # GRU 模型指标
        gru_path = self.output_dir / "gru_metrics.csv"
        if gru_path.exists():
            metrics['GRU'] = pd.read_csv(gru_path)
            print(f"Loaded GRU metrics")
        
        return metrics
    
    def load_merged_data(self):
        """加载并合并所有数据"""
        train = self.load_train_data()
        test = self.load_test_data()
        predictions = self.load_predictions()
        stores = self.load_stores()
        
        if test is not None and predictions is not None:
            # 合合测试数据和预测结果
            test_merged = test.merge(predictions, on='id', how='left')
            test_merged.rename(columns={'sales': 'predicted_sales'}, inplace=True)
            
            if stores is not None:
                test_merged = test_merged.merge(stores, on='store_nbr', how='left')
            
            return train, test_merged, stores
        
        return train, None, stores


class ModelEvaluator:
    """模型评估器 - 计算和展示评估指标"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """计算评估指标"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 确保预测值非负
        y_pred = np.maximum(y_pred, 0)
        y_true = np.maximum(y_true, 0)
        
        # 计算各项指标
        try:
            rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
        except:
            rmsle = np.nan
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan
        
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'RMSLE': round(rmsle, 4) if not np.isnan(rmsle) else 'N/A',
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4) if not np.isnan(r2) else 'N/A',
            'MAPE': round(mape, 2)
        }
    
    @staticmethod
    def get_validation_metrics(metrics_dict):
        """获取验证集指标"""
        val_metrics = {}
        for model, df in metrics_dict.items():
            if 'split' in df.columns:
                val_row = df[df['split'] == 'validation']
                if len(val_row) > 0:
                    val_metrics[model] = {
                        'RMSLE': val_row['RMSLE'].values[0],
                        'MAE': val_row['MAE'].values[0],
                        'RMSE': val_row['RMSE'].values[0]
                    }
            elif 'rmsle' in df.columns:
                # Baseline 格式
                val_metrics[model] = {
                    'RMSLE': df['rmsle'].mean(),
                    'MAE': df['mae'].mean(),
                    'RMSE': df['rmse'].mean()
                }
        return val_metrics


class VisualizationEngine:
    """可视化引擎 - 生成多样化图表"""
    
    def __init__(self, figures_dir=None, interactive_dir=None):
        self.figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
        self.interactive_dir = Path(interactive_dir) if interactive_dir else INTERACTIVE_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 折线图 ====================
    
    def plot_sales_history(self, train_data, store_nbr=None, family=None, 
                           date_range=None, save_path=None):
        """绘制销售历史折线图"""
        df = train_data.copy()
        
        # 数据筛选
        if store_nbr is not None:
            df = df[df['store_nbr'] == store_nbr]
        if family is not None:
            df = df[df['family'] == family]
        if date_range is not None:
            df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
        
        if df.empty:
            print("No data after filtering")
            return
        
        # 按日期汇总
        daily_sales = df.groupby('date')['sales'].sum().reset_index()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(daily_sales['date'], daily_sales['sales'], 
                color='steelblue', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales', fontsize=12)
        ax.set_title('Historical Sales Trend', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 设置日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_prediction_comparison(self, dates, y_true, y_pred, model_name,
                                   confidence_interval=None, save_path=None):
        """绘制预测对比折线图（含置信区间）"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 绘制真实值
        ax.plot(dates, y_true, label='Actual Sales', 
                color='steelblue', linewidth=2, alpha=0.8)
        
        # 绘制预测值
        ax.plot(dates, y_pred, label=f'{model_name} Prediction', 
                color='coral', linewidth=2, alpha=0.8, linestyle='--')
        
        # 绘制置信区间
        if confidence_interval is not None:
            ax.fill_between(dates, 
                           y_pred - confidence_interval,
                           y_pred + confidence_interval,
                           color='coral', alpha=0.2, label='95% Confidence Interval')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales', fontsize=12)
        ax.set_title(f'{model_name}: Prediction vs Actual', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_predicted_sales_timeseries(self, test_merged, train_data=None, save_path=None):
        """绘制预测销售量时间序列趋势图"""
        if test_merged is None:
            print("No prediction data available")
            return
        
        # 按日期汇总预测销售量
        daily_pred = test_merged.groupby('date')['predicted_sales'].sum().reset_index()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 绘制预测销售量趋势
        ax.plot(daily_pred['date'], daily_pred['predicted_sales'], 
                color='coral', linewidth=2, alpha=0.8, linestyle='--',
                label='Predicted Sales')
        
        # 如果有训练数据，绘制历史销售趋势作为对比
        if train_data is not None:
            # 获取最近30天的历史销售数据
            last_train_date = train_data['date'].max()
            recent_train = train_data[train_data['date'] > last_train_date - pd.Timedelta(days=30)]
            daily_hist = recent_train.groupby('date')['sales'].sum().reset_index()
            ax.plot(daily_hist['date'], daily_hist['sales'], 
                    color='steelblue', linewidth=2, alpha=0.8,
                    label='Historical Sales (Last 30 days)')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales', fontsize=12)
        ax.set_title('Predicted Sales Time Series', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_predicted_sales_distribution(self, test_merged, save_path=None):
        """绘制预测销售量分布图（直方图+箱线图）"""
        if test_merged is None:
            print("No prediction data available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1 = axes[0]
        sns.histplot(test_merged['predicted_sales'], bins=50, kde=True, ax=ax1, color='coral')
        ax1.axvline(x=test_merged['predicted_sales'].mean(), color='green', linestyle='--', 
                    lw=2, label=f'Mean: {test_merged["predicted_sales"].mean():.2f}')
        ax1.set_xlabel('Predicted Sales', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Predicted Sales Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱线图（按商店类型）
        ax2 = axes[1]
        if 'type' in test_merged.columns:
            sns.boxplot(x='type', y='predicted_sales', data=test_merged, ax=ax2,
                       palette='Set2')
            ax2.set_xlabel('Store Type', fontsize=11)
            ax2.set_ylabel('Predicted Sales', fontsize=11)
            ax2.set_title('Predicted Sales by Store Type', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_predicted_sales_by_family(self, test_merged, top_n=10, save_path=None):
        """绘制各商品类别预测销售量对比图"""
        if test_merged is None:
            print("No prediction data available")
            return
        
        family_pred = test_merged.groupby('family')['predicted_sales'].sum().sort_values(ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(family_pred)))
        
        bars = ax.barh(family_pred.index, family_pred.values, color=colors)
        
        ax.set_xlabel('Total Predicted Sales', fontsize=12)
        ax.set_ylabel('Product Family', fontsize=12)
        ax.set_title(f'Top {top_n} Product Families by Predicted Sales', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    # ==================== 柱状图 ====================
    
    def plot_model_comparison_bar(self, metrics_dict, save_path=None):
        """绘制模型性能对比柱状图"""
        models = list(metrics_dict.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics_names = ['RMSLE', 'MAE', 'RMSE', 'MAPE']
        colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
        
        for idx, (metric, color) in enumerate(zip(metrics_names, colors)):
            ax = axes[idx]
            values = []
            for m in models:
                val = metrics_dict[m].get(metric, 0)
                if isinstance(val, str):
                    val = 0
                values.append(val)
            
            bars = ax.bar(models, values, color=color, alpha=0.8, edgecolor='white')
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_store_sales_bar(self, data, top_n=20, save_path=None):
        """绘制商店销售额柱状图"""
        store_sales = data.groupby('store_nbr')['sales'].sum().sort_values(ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(store_sales)))
        
        bars = ax.bar(store_sales.index.astype(str), store_sales.values, color=colors)
        
        ax.set_xlabel('Store Number', fontsize=12)
        ax.set_ylabel('Total Sales', fontsize=12)
        ax.set_title(f'Top {top_n} Stores by Sales', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_family_sales_bar(self, data, top_n=15, save_path=None):
        """绘制商品类别销售额柱状图"""
        family_sales = data.groupby('family')['sales'].sum().sort_values(ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(family_sales)))
        
        bars = ax.barh(family_sales.index, family_sales.values, color=colors)
        
        ax.set_xlabel('Total Sales', fontsize=12)
        ax.set_ylabel('Product Family', fontsize=12)
        ax.set_title(f'Top {top_n} Product Families by Sales', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    # ==================== 热力图 ====================
    
    def plot_store_family_heatmap(self, data, stores=None, families=None, 
                                   value_col='sales', save_path=None):
        """绘制商店-商品类别热力图"""
        # 数据聚合
        pivot_data = data.groupby(['store_nbr', 'family'])[value_col].mean().reset_index()
        pivot_table = pivot_data.pivot(index='store_nbr', columns='family', values=value_col)
        
        # 筛选
        if stores is not None:
            pivot_table = pivot_table.loc[pivot_table.index.isin(stores)]
        if families is not None:
            pivot_table = pivot_table[families]
        
        # 选择Top商品类别
        top_families = pivot_table.sum().sort_values(ascending=False).head(10).index
        pivot_table = pivot_table[top_families]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': value_col})
        
        ax.set_title(f'Store-Family {value_col} Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Product Family', fontsize=12)
        ax.set_ylabel('Store Number', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_metrics_heatmap(self, baseline_data, save_path=None):
        """绘制基线模型指标热力图"""
        # 选择Top商品类别
        top_families = baseline_data.groupby('family')['rmsle'].mean().sort_values().head(15).index
        filtered = baseline_data[baseline_data['family'].isin(top_families)]
        
        pivot = filtered.pivot(index='store_nbr', columns='family', values='rmsle')
        pivot = pivot.head(15)  # 限制商店数量
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   ax=ax, cbar_kws={'label': 'RMSLE'})
        
        ax.set_title('Baseline RMSLE by Store and Family', fontsize=14, fontweight='bold')
        ax.set_xlabel('Product Family', fontsize=12)
        ax.set_ylabel('Store Number', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    # ==================== 散点图与残差分析 ====================
    
    def plot_prediction_scatter(self, y_true, y_pred, model_name, save_path=None):
        """绘制预测值 vs 真实值散点图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 散点图
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=20, c='steelblue')
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Sales', fontsize=11)
        ax1.set_ylabel('Predicted Sales', fontsize=11)
        ax1.set_title(f'{model_name}: Predicted vs Actual', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 残差图
        ax2 = axes[1]
        residuals = y_pred - y_true
        ax2.scatter(y_true, residuals, alpha=0.5, s=20, c='coral')
        ax2.axhline(y=0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel('Actual Sales', fontsize=11)
        ax2.set_ylabel('Residuals (Predicted - Actual)', fontsize=11)
        ax2.set_title(f'{model_name}: Residual Analysis', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_residual_distribution(self, y_true, y_pred, model_name, save_path=None):
        """绘制残差分布图"""
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1 = axes[0]
        sns.histplot(residuals, bins=50, kde=True, ax=ax1, color='steelblue')
        ax1.axvline(x=0, color='red', linestyle='--', lw=2)
        ax1.axvline(x=residuals.mean(), color='green', linestyle='--', lw=2, label=f'Mean: {residuals.mean():.2f}')
        ax1.set_xlabel('Residuals', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'{model_name}: Residual Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q图
        ax2 = axes[1]
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'{model_name}: Q-Q Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    # ==================== 雷达图 ====================
    
    def plot_metrics_radar(self, metrics_dict, save_path=None):
        """绘制雷达图对比各模型"""
        models = list(metrics_dict.keys())
        metrics_names = ['RMSLE', 'MAE', 'RMSE', 'R2']
        
        # 准备数据
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for (model, metrics), color in zip(metrics_dict.items(), colors):
            values = []
            for m in metrics_names:
                val = metrics.get(m, 0)
                if isinstance(val, str):
                    val = 0
                values.append(val)
            
            # 归一化
            max_vals = []
            for met in metrics_names:
                max_val = 0
                for mod in models:
                    v = metrics_dict[mod].get(met, 0)
                    if isinstance(v, (int, float)) and v > max_val:
                        max_val = v
                max_vals.append(max_val if max_val > 0 else 1)
            
            values_norm = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]
            values_norm += values_norm[:1]
            
            ax.plot(angles, values_norm, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values_norm, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names, fontsize=11)
        ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    # ==================== 箱线图 ====================
    
    def plot_sales_boxplot(self, data, group_by='family', top_n=10, save_path=None):
        """绘制销售额箱线图"""
        # 选择Top类别
        top_groups = data.groupby(group_by)['sales'].sum().sort_values(ascending=False).head(top_n).index
        filtered = data[data[group_by].isin(top_groups)]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        filtered.boxplot(column='sales', by=group_by, ax=ax)
        
        ax.set_xlabel(group_by, fontsize=12)
        ax.set_ylabel('Sales', fontsize=12)
        ax.set_title(f'Sales Distribution by {group_by}', fontsize=14, fontweight='bold')
        plt.suptitle('')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    # ==================== 交互式可视化 ====================
    
    def create_interactive_dashboard(self, train_data, test_merged, metrics_dict):
        """创建交互式仪表板（使用Plotly）"""
        if not INTERACTIVE_AVAILABLE:
            print("Plotly not available, skipping interactive dashboard")
            return
        
        # 1. 交互式销售历史图
        daily_sales = train_data.groupby('date')['sales'].sum().reset_index()
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=daily_sales['date'],
            y=daily_sales['sales'],
            mode='lines',
            name='Sales',
            line=dict(color='steelblue', width=1.5)
        ))
        fig1.update_layout(
            title='Interactive Sales History',
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        fig1.write_html(self.interactive_dir / "sales_history_interactive.html")
        print(f"Saved interactive: sales_history_interactive.html")
        
        # 2. 交互式模型对比图
        models = list(metrics_dict.keys())
        metrics_names = ['RMSLE', 'MAE', 'RMSE']
        
        fig2 = make_subplots(rows=1, cols=3, subplot_titles=metrics_names)
        
        for idx, metric in enumerate(metrics_names):
            values = []
            for m in models:
                val = metrics_dict[m].get(metric, 0)
                if isinstance(val, str):
                    val = 0
                values.append(val)
            
            fig2.add_trace(
                go.Bar(x=models, y=values, name=metric),
                row=1, col=idx+1
            )
        
        fig2.update_layout(
            title_text='Interactive Model Comparison',
            showlegend=False
        )
        fig2.write_html(self.interactive_dir / "model_comparison_interactive.html")
        print(f"Saved interactive: model_comparison_interactive.html")
        
        # 3. 交互式热力图
        if test_merged is not None and 'predicted_sales' in test_merged.columns:
            pivot = test_merged.groupby(['store_nbr', 'family'])['predicted_sales'].mean().reset_index()
            pivot_table = pivot.pivot(index='store_nbr', columns='family', values='predicted_sales')
            
            # 选择Top类别
            top_families = pivot_table.sum().sort_values(ascending=False).head(10).index
            pivot_table = pivot_table[top_families].head(20)
            
            fig3 = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='YlOrRd'
            ))
            fig3.update_layout(
                title='Interactive Store-Family Prediction Heatmap',
                xaxis_title='Product Family',
                yaxis_title='Store Number'
            )
            fig3.write_html(self.interactive_dir / "heatmap_interactive.html")
            print(f"Saved interactive: heatmap_interactive.html")


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    
    def generate_evaluation_report(self, metrics_dict, baseline_data=None):
        """生成评估报告"""
        report = "# 销售预测模型评估报告\n\n"
        report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 1. 项目概述\n\n"
        report += "本报告基于真实实验数据对多个销售预测模型进行全面评估。\n\n"
        report += "### 评估模型\n"
        report += "- **XGBoost**: 梯度提升树模型\n"
        report += "- **LSTM**: 长短期记忆网络\n"
        report += "- **GRU**: 门控循环单元\n\n"
        
        report += "## 2. 评估指标说明\n\n"
        report += "| 指标 | 英文名 | 说明 | 方向 |\n"
        report += "|------|--------|------|------|\n"
        report += "| RMSLE | Root Mean Squared Log Error | 均方根对数误差 | ↓ 越小越好 |\n"
        report += "| MAE | Mean Absolute Error | 平均绝对误差 | ↓ 越小越好 |\n"
        report += "| RMSE | Root Mean Squared Error | 均方根误差 | ↓ 越小越好 |\n"
        report += "| MAPE | Mean Absolute Percentage Error | 平均绝对百分比误差 | ↓ 越小越好 |\n"
        report += "| R² | R-squared | 决定系数 | ↑ 越大越好 |\n\n"
        
        report += "## 3. 模型性能对比\n\n"
        report += "| 模型 | RMSLE | MAE | RMSE |\n"
        report += "|------|-------|-----|------|\n"
        
        for model, metrics in metrics_dict.items():
            rmsle = metrics.get('RMSLE', 'N/A')
            mae = metrics.get('MAE', 'N/A')
            rmse = metrics.get('RMSE', 'N/A')
            
            if isinstance(rmsle, float):
                rmsle = f"{rmsle:.4f}"
            if isinstance(mae, float):
                mae = f"{mae:.2f}"
            if isinstance(rmse, float):
                rmse = f"{rmse:.2f}"
            
            report += f"| {model} | {rmsle} | {mae} | {rmse} |\n"
        
        report += "\n## 4. 分析结论\n\n"
        
        # 找出最佳模型
        best_rmsle_model = None
        best_rmsle_val = float('inf')
        
        for model, metrics in metrics_dict.items():
            val = metrics.get('RMSLE', float('inf'))
            if isinstance(val, (int, float)) and val < best_rmsle_val:
                best_rmsle_val = val
                best_rmsle_model = model
        
        if best_rmsle_model:
            report += f"### 最佳模型\n\n"
            report += f"**{best_rmsle_model}** 在 RMSLE 指标上表现最佳（{best_rmsle_val:.4f}）。\n\n"
        
        # Baseline 详细分析（作为参考基准）
        if baseline_data is not None and not baseline_data.empty:
            report += "## 5. Baseline 参考基准\n\n"
            report += "Baseline（Seasonal Naive）作为参考基准，用于对比模型性能。\n\n"
            report += f"- 评估记录数: {len(baseline_data)}\n"
            report += f"- 平均 RMSLE: {baseline_data['rmsle'].mean():.4f}\n"
            report += f"- 平均 MAE: {baseline_data['mae'].mean():.2f}\n"
            report += f"- 平均 RMSE: {baseline_data['rmse'].mean():.2f}\n\n"
            
            # Top/Bottom 商品类别
            family_rmsle = baseline_data.groupby('family')['rmsle'].mean()
            best_families = family_rmsle.sort_values().head(5)
            worst_families = family_rmsle.sort_values().tail(5)
            
            report += "### 预测效果最好的商品类别\n\n"
            for fam, val in best_families.items():
                report += f"- {fam}: {val:.4f}\n"
            
            report += "\n### 预测效果最差的商品类别\n\n"
            for fam, val in worst_families.items():
                report += f"- {fam}: {val:.4f}\n"
        
        report += "\n## 6. 可视化图表清单\n\n"
        report += "### 静态图表\n"
        report += "- `sales_history.png` - 销售历史折线图\n"
        report += "- `model_comparison.png` - 模型性能对比柱状图\n"
        report += "- `metrics_radar.png` - 模型性能雷达图\n"
        report += "- `store_sales_bar.png` - 商店销售额柱状图\n"
        report += "- `family_sales_bar.png` - 商品类别销售额柱状图\n"
        report += "- `store_family_heatmap.png` - 商店-商品热力图\n"
        report += "- `metrics_heatmap.png` - 模型指标热力图\n"
        report += "- `sales_boxplot.png` - 销售分布箱线图\n"
        report += "- `prediction_scatter.png` - 预测值与真实值散点图\n"
        report += "- `residual_distribution.png` - 残差分布图\n"
        report += "- `predicted_sales_timeseries.png` - 预测销售量时间序列图\n"
        report += "- `predicted_sales_distribution.png` - 预测销售量分布图\n"
        report += "- `predicted_sales_by_family.png` - 各商品类别预测销售量对比图\n\n"
        
        if INTERACTIVE_AVAILABLE:
            report += "### 交互式图表\n"
            report += "- `sales_history_interactive.html` - 交互式销售历史\n"
            report += "- `model_comparison_interactive.html` - 交互式模型对比\n"
            report += "- `heatmap_interactive.html` - 交互式热力图\n\n"
        
        report += "---\n"
        report += "*本报告使用真实实验数据生成*\n"
        
        # 保存报告
        report_path = self.output_dir / "evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved: {report_path}")
        return report


def run_visualization(data_dir=None, output_dir=None, use_synthetic=False, **kwargs):
    """
    可视化入口函数（兼容 train_all.py 调用）
    
    Args:
        data_dir: 数据目录路径（支持相对路径如 "." 或 ".."）
        output_dir: 输出目录路径
        use_synthetic: 是否使用合成数据（当前版本忽略，始终使用真实数据）
        **kwargs: 其他参数
    
    Returns:
        模型评估指标字典
    """
    print("\n" + "=" * 60)
    print("Enhanced Visualization - Using Real Experimental Data")
    print("=" * 60)
    
    # 处理相对路径 - train_all.py 调用时传入 Path(".")
    if data_dir is not None:
        data_dir = Path(data_dir).resolve()
        # 如果是项目根目录，使用默认的 data 和 output 目录
        project_root = PROJECT_ROOT
        if data_dir == project_root or data_dir.name == "store-sales-time-series-forecasting":
            data_dir = None  # 使用默认 DATA_DIR
            output_dir = None  # 使用默认 OUTPUT_DIR
    
    # 调用完整可视化流程
    val_metrics = run_full_visualization(
        data_dir=data_dir,
        output_dir=output_dir,
        store_filter=None,
        family_filter=None,
        date_range=None
    )
    
    print("\n" + "=" * 60)
    print("Visualization completed successfully!")
    print("=" * 60)
    
    return val_metrics


def run_full_visualization(data_dir=None, output_dir=None, 
                           store_filter=None, family_filter=None,
                           date_range=None):
    """运行完整可视化分析"""
    print("=" * 60)
    print("销售预测模型评估与可视化分析")
    print("使用真实实验数据")
    print("=" * 60)
    
    # 处理默认路径
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    
    # 初始化组件
    loader = SalesDataLoader(data_dir, output_dir)
    viz = VisualizationEngine(output_dir / "figures", output_dir / "interactive")
    report_gen = ReportGenerator(output_dir)
    
    # 加载真实数据
    print("\n[1] 加载真实数据...")
    train_data, test_merged, stores = loader.load_merged_data()
    metrics_data = loader.load_model_metrics()
    
    if train_data is None:
        print("Error: Cannot load train data")
        return
    
    # 获取验证集指标
    print("\n[2] 提取评估指标...")
    val_metrics = ModelEvaluator.get_validation_metrics(metrics_data)
    
    if val_metrics:
        print("\n模型验证集指标:")
        for model, metrics in val_metrics.items():
            print(f"  {model}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    # 生成可视化图表
    print("\n[3] 生成可视化图表...")
    
    figures_dir = viz.figures_dir
    
    # 3.1 销售历史折线图
    viz.plot_sales_history(
        train_data, 
        store_nbr=store_filter,
        family=family_filter,
        date_range=date_range,
        save_path=figures_dir / "sales_history.png"
    )
    
    # 3.2 模型性能对比柱状图（排除Baseline，只对比XGBoost、LSTM、GRU）
    if val_metrics:
        # 过滤掉Baseline，只保留机器学习/深度学习模型
        ml_metrics = {k: v for k, v in val_metrics.items() if k != 'Baseline'}
        if ml_metrics:
            viz.plot_model_comparison_bar(
                ml_metrics,
                save_path=figures_dir / "model_comparison.png"
            )
            
            # 3.3 雷达图
            viz.plot_metrics_radar(
                ml_metrics,
                save_path=figures_dir / "metrics_radar.png"
            )
    
    # 3.4 商店销售额柱状图
    viz.plot_store_sales_bar(
        train_data,
        top_n=20,
        save_path=figures_dir / "store_sales_bar.png"
    )
    
    # 3.5 商品类别销售额柱状图
    viz.plot_family_sales_bar(
        train_data,
        top_n=15,
        save_path=figures_dir / "family_sales_bar.png"
    )
    
    # 3.6 商店-商品热力图
    viz.plot_store_family_heatmap(
        train_data,
        save_path=figures_dir / "store_family_heatmap.png"
    )
    
    # 3.7 Baseline指标热力图
    if 'Baseline' in metrics_data:
        viz.plot_metrics_heatmap(
            metrics_data['Baseline'],
            save_path=figures_dir / "metrics_heatmap.png"
        )
    
    # 3.8 销售分布箱线图
    viz.plot_sales_boxplot(
        train_data,
        group_by='family',
        top_n=10,
        save_path=figures_dir / "sales_boxplot.png"
    )
    
    # 3.9 预测散点图（如果有预测数据）
    if test_merged is not None and 'predicted_sales' in test_merged.columns:
        # 获取对应的历史数据作为真实值参考
        last_train_date = train_data['date'].max()
        comparison_train = train_data[train_data['date'] > last_train_date - pd.Timedelta(days=31)]
        
        if not comparison_train.empty:
            y_true_sample = comparison_train['sales'].values[:len(test_merged)]
            y_pred_sample = test_merged['predicted_sales'].values[:len(y_true_sample)]
            
            if len(y_true_sample) > 0 and len(y_pred_sample) > 0:
                viz.plot_prediction_scatter(
                    y_true_sample, y_pred_sample,
                    "Model Prediction",
                    save_path=figures_dir / "prediction_scatter.png"
                )
                
                viz.plot_residual_distribution(
                    y_true_sample, y_pred_sample,
                    "Model Prediction",
                    save_path=figures_dir / "residual_distribution.png"
                )
            
            # 3.9.4 预测销售量时间序列图
            viz.plot_predicted_sales_timeseries(
                test_merged,
                train_data=train_data,
                save_path=figures_dir / "predicted_sales_timeseries.png"
            )
            
            # 3.9.5 预测销售量分布图
            viz.plot_predicted_sales_distribution(
                test_merged,
                save_path=figures_dir / "predicted_sales_distribution.png"
            )
            
            # 3.9.6 各商品类别预测销售量对比图
            viz.plot_predicted_sales_by_family(
                test_merged,
                top_n=10,
                save_path=figures_dir / "predicted_sales_by_family.png"
            )
    
    # 3.10 交互式仪表板（只对比 XGBoost、LSTM、GRU）
    print("\n[4] 生成交互式可视化...")
    ml_metrics = {k: v for k, v in val_metrics.items() if k != 'Baseline'} if val_metrics else {}
    viz.create_interactive_dashboard(train_data, test_merged, ml_metrics)
    
    # 生成评估报告
    print("\n[5] 生成评估报告...")
    baseline_data = metrics_data.get('Baseline', None)
    # 报告中只对比 XGBoost、LSTM、GRU
    ml_metrics = {k: v for k, v in val_metrics.items() if k != 'Baseline'} if val_metrics else {}
    report_gen.generate_evaluation_report(ml_metrics, baseline_data)
    
    print("\n" + "=" * 60)
    print("可视化分析完成！")
    print(f"静态图表目录: {figures_dir}")
    print(f"交互图表目录: {viz.interactive_dir}")
    print(f"评估报告: {output_dir / 'evaluation_report.md' if output_dir else OUTPUT_DIR / 'evaluation_report.md'}")
    print("=" * 60)
    
    return val_metrics


def main():
    parser = argparse.ArgumentParser(description="销售预测模型评估与可视化（使用真实数据）")
    parser.add_argument("--data-dir", type=Path, default=None, help="数据目录")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录")
    parser.add_argument("--store", type=int, default=None, help="筛选商店编号")
    parser.add_argument("--family", type=str, default=None, help="筛选商品类别")
    parser.add_argument("--start-date", type=str, default=None, help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="结束日期 (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # 处理日期范围
    date_range = None
    if args.start_date and args.end_date:
        date_range = [pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)]
    
    # 运行可视化
    metrics = run_full_visualization(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        store_filter=args.store,
        family_filter=args.family,
        date_range=date_range
    )
    
    return metrics


if __name__ == "__main__":
    metrics = main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
import os

def calculate_metrics(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'RMSLE': round(rmsle, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4)
    }

def plot_predictions_vs_actual(y_true, y_pred, title, output_path):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_residuals(y_true, y_pred, title, output_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(importance_df, title, output_path, top_n=20):
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_metrics_comparison(metrics_dict, output_path):
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
    
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', x='Model', y=['RMSLE', 'MAE', 'RMSE'], figsize=(12, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_comparison_report(models_metrics, output_path):
    report = "# Model Comparison Report\n\n"
    report += "## Overview\n"
    report += "This report compares the performance of different forecasting models.\n\n"
    
    report += "## Performance Metrics\n"
    report += "| Model | RMSLE | MAE | RMSE |\n"
    report += "|-------|-------|-----|------|\n"
    
    for model_name, metrics in models_metrics.items():
        report += f"| {model_name} | {metrics['RMSLE']} | {metrics['MAE']} | {metrics['RMSE']} |\n"
    
    report += "\n## Analysis\n"
    report += "### Best Performing Model\n"
    best_model = min(models_metrics.keys(), key=lambda x: models_metrics[x]['RMSLE'])
    report += f"The best performing model based on RMSLE is **{best_model}**.\n\n"
    
    report += "### Key Observations\n"
    report += "- XGBoost generally performs well with structured tabular data and engineered features.\n"
    report += "- LSTM/GRU models capture sequential patterns but may require more tuning.\n"
    report += "- Feature engineering plays a crucial role in model performance.\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report

class ModelEvaluator:
    def __init__(self):
        self.metrics_history = {}
    
    def evaluate_model(self, model_name, y_true, y_pred):
        metrics = calculate_metrics(y_true, y_pred)
        self.metrics_history[model_name] = metrics
        return metrics
    
    def add_metrics(self, model_name, metrics):
        self.metrics_history[model_name] = metrics
    
    def plot_comparison(self, output_dir='outputs/figures'):
        os.makedirs(output_dir, exist_ok=True)
        
        plot_metrics_comparison(self.metrics_history, os.path.join(output_dir, 'model_comparison.png'))
        
        return self.metrics_history
    
    def generate_report(self, output_path='outputs/model_comparison_report.md'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return generate_comparison_report(self.metrics_history, output_path)
    
    def get_best_model(self):
        if not self.metrics_history:
            return None
        
        best_model = min(self.metrics_history.keys(), key=lambda x: self.metrics_history[x]['RMSLE'])
        return best_model, self.metrics_history[best_model]
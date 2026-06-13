import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

from .model import XGBoostModel

logger = logging.getLogger(__name__)

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    df['sales_lag_1'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1)
    df['sales_lag_7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
    df['sales_lag_14'] = df.groupby(['store_nbr', 'family'])['sales'].shift(14)
    df['sales_lag_28'] = df.groupby(['store_nbr', 'family'])['sales'].shift(28)
    
    df['rolling_mean_7'] = df.groupby(['store_nbr', 'family'])['sales'].rolling(7).mean().reset_index(level=[0,1], drop=True)
    df['rolling_std_7'] = df.groupby(['store_nbr', 'family'])['sales'].rolling(7).std().reset_index(level=[0,1], drop=True)
    df['rolling_mean_14'] = df.groupby(['store_nbr', 'family'])['sales'].rolling(14).mean().reset_index(level=[0,1], drop=True)
    
    df = df.dropna()
    
    df = pd.get_dummies(df, columns=['store_nbr', 'family', 'dayofweek', 'month'])
    
    return df

def train_xgboost_model(train_filepath, model_output_path, log_filepath):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Loading and preparing data...")
    df = prepare_data(train_filepath)
    
    features = [col for col in df.columns if col not in ['date', 'sales', 'id']]
    X = df[features]
    y = df['sales']
    
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 10,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42
    }
    
    logger.info("Training XGBoost model...")
    model = XGBoostModel(params)
    model.train(X_train, y_train, X_val, y_val, verbose=True)
    
    logger.info("Evaluating model...")
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    
    logger.info(f"Training metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")
    
    logger.info("Saving model...")
    model.save_model(model_output_path)
    
    feature_importance = model.get_feature_importance(features)
    feature_importance_path = model_output_path.replace('.pkl', '_feature_importance.csv')
    feature_importance.to_csv(feature_importance_path, index=False)
    logger.info(f"Feature importance saved to {feature_importance_path}")
    
    return model, train_metrics, val_metrics

if __name__ == '__main__':
    train_filepath = 'data/train.csv'
    model_output_path = 'outputs/models/xgboost_model.pkl'
    log_filepath = 'logs/xgboost_training.log'
    
    train_xgboost_model(train_filepath, model_output_path, log_filepath)
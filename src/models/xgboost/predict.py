import pandas as pd
import numpy as np
import logging
import os

from .model import XGBoostModel

logger = logging.getLogger(__name__)

def prepare_test_data(filepath, train_df=None):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    if train_df is not None:
        df = pd.concat([train_df, df], ignore_index=True)
    
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

def predict_xgboost_model(model_path, test_filepath, output_path, train_filepath=None):
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Loading model...")
    model = XGBoostModel()
    model.load_model(model_path)
    
    logger.info("Loading and preparing test data...")
    train_df = None
    if train_filepath:
        train_df = pd.read_csv(train_filepath)
        train_df['date'] = pd.to_datetime(train_df['date'])
    
    test_df = prepare_test_data(test_filepath, train_df)
    
    features = [col for col in test_df.columns if col not in ['date', 'sales', 'id']]
    
    if 'sales' in test_df.columns:
        X = test_df[features]
        y = test_df['sales']
    else:
        X = test_df[features]
        y = None
    
    logger.info(f"Test data shape: {X.shape}")
    
    logger.info("Making predictions...")
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)
    
    test_df['predicted_sales'] = predictions
    
    if y is not None:
        from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
        rmsle = np.sqrt(mean_squared_log_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        logger.info(f"Test metrics - RMSLE: {rmsle:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    logger.info(f"Saving predictions to {output_path}")
    test_df[['date', 'store_nbr', 'family', 'predicted_sales']].to_csv(output_path, index=False)
    
    return test_df

if __name__ == '__main__':
    model_path = 'outputs/models/xgboost_model.pkl'
    test_filepath = 'data/test.csv'
    output_path = 'outputs/predictions/xgboost_predictions.csv'
    
    predict_xgboost_model(model_path, test_filepath, output_path)
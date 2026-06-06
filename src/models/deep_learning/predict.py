import pandas as pd
import numpy as np
import logging
import os

from .model import DeepLearningModel

logger = logging.getLogger(__name__)

def prepare_test_data(filepath, train_df=None):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    if 'sales' not in df.columns:
        df['sales'] = 0
    
    if train_df is not None:
        df = pd.concat([train_df, df], ignore_index=True)
    
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    df['sales_lag_1'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1)
    df['sales_lag_7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
    df['sales_lag_14'] = df.groupby(['store_nbr', 'family'])['sales'].shift(14)
    
    df['rolling_mean_7'] = df.groupby(['store_nbr', 'family'])['sales'].rolling(7).mean().reset_index(level=[0,1], drop=True)
    
    df = df.dropna()
    
    store_dummies = pd.get_dummies(df['store_nbr'], prefix='store')
    family_dummies = pd.get_dummies(df['family'], prefix='family')
    
    df = pd.concat([df, store_dummies, family_dummies], axis=1)
    
    return df

def predict_deep_learning_model(model_type, model_path, test_filepath, output_path, train_filepath=None):
    logging.basicConfig(level=logging.INFO)
    
    logger.info(f"Loading {model_type.upper()} model...")
    model = DeepLearningModel(model_type=model_type)
    model.load_model(model_path)
    
    logger.info("Loading and preparing test data...")
    train_df = None
    if train_filepath:
        train_df = pd.read_csv(train_filepath)
        train_df['date'] = pd.to_datetime(train_df['date'])
    
    test_df = prepare_test_data(test_filepath, train_df)
    
    features = [col for col in test_df.columns if col not in ['date', 'sales', 'id', 'store_nbr', 'family']]
    X = test_df[features]
    
    has_sales = 'sales' in test_df.columns and test_df['sales'].sum() > 0
    if has_sales:
        y = test_df['sales']
    else:
        y = None
    
    logger.info(f"Test data shape: {X.shape}")
    
    logger.info("Making predictions...")
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)
    
    seq_len = model.params['sequence_length']
    test_df = test_df.iloc[seq_len:]
    test_df['predicted_sales'] = predictions
    
    if y is not None:
        y_eval = y.iloc[seq_len:].values
        from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
        rmsle = np.sqrt(mean_squared_log_error(y_eval, predictions))
        mae = mean_absolute_error(y_eval, predictions)
        rmse = np.sqrt(mean_squared_error(y_eval, predictions))
        logger.info(f"Test metrics - RMSLE: {rmsle:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    logger.info(f"Saving predictions to {output_path}")
    test_df[['date', 'store_nbr', 'family', 'predicted_sales']].to_csv(output_path, index=False)
    
    return test_df

if __name__ == '__main__':
    model_type = 'lstm'
    model_path = f'outputs/models/{model_type}_model.h5'
    test_filepath = 'data/test.csv'
    output_path = f'outputs/predictions/{model_type}_predictions.csv'
    
    predict_deep_learning_model(model_type, model_path, test_filepath, output_path)
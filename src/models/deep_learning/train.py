import pandas as pd
import numpy as np
import logging
import os

from .model import DeepLearningModel

logger = logging.getLogger(__name__)

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
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

def train_deep_learning_model(model_type, train_filepath, model_output_path, log_filepath):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Loading and preparing data for {model_type.upper()} model...")
    df = prepare_data(train_filepath)
    
    features = [col for col in df.columns if col not in ['date', 'sales', 'id', 'store_nbr', 'family']]
    X = df[features]
    y = df['sales']
    
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    
    params = {
        'units': 128,
        'layers': 2,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'sequence_length': 28,
        'epochs': 50,
        'batch_size': 64
    }
    
    logger.info(f"Training {model_type.upper()} model with params: {params}")
    model = DeepLearningModel(model_type=model_type, params=params)
    
    try:
        history = model.train(X_train, y_train, X_val, y_val)
        
        logger.info("Evaluating model...")
        train_metrics = model.evaluate(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)
        
        logger.info(f"Training metrics: {train_metrics}")
        logger.info(f"Validation metrics: {val_metrics}")
        
        logger.info(f"Saving {model_type.upper()} model...")
        model.save_model(model_output_path)
        
        return model, train_metrics, val_metrics, history
    except Exception as e:
        logger.error(f"Error training {model_type.upper()} model: {e}")
        raise

if __name__ == '__main__':
    model_type = 'lstm'
    train_filepath = 'data/train.csv'
    model_output_path = f'outputs/models/{model_type}_model.h5'
    log_filepath = f'logs/{model_type}_training.log'
    
    train_deep_learning_model(model_type, train_filepath, model_output_path, log_filepath)
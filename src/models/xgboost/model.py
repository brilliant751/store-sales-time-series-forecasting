import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        watchlist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist.append((dval, 'val'))
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 100),
            evals=watchlist,
            verbose_eval=verbose
        )
        return self.model
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        predictions = np.maximum(predictions, 0)
        
        rmsle = np.sqrt(mean_squared_log_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        return {
            'RMSLE': rmsle,
            'MAE': mae,
            'RMSE': rmse
        }
    
    def get_feature_importance(self, feature_names):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        importance = self.model.get_score(importance_type='weight')
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        all_features = pd.DataFrame({'feature': feature_names})
        importance_df = all_features.merge(importance_df, on='feature', how='left').fillna(0)
        
        return importance_df
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        pickle.dump(self.model, open(filepath, 'wb'))
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = pickle.load(open(filepath, 'rb'))
        logger.info(f"Model loaded from {filepath}")
        return self
    
    @staticmethod
    def hyperparameter_tuning(X, y, param_grid, n_splits=5):
        best_params = None
        best_score = float('inf')
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for params in param_grid:
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = XGBoostModel(params)
                model.train(X_train, y_train, X_val, y_val, verbose=False)
                predictions = model.predict(X_val)
                predictions = np.maximum(predictions, 0)
                rmsle = np.sqrt(mean_squared_log_error(y_val, predictions))
                scores.append(rmsle)
            
            mean_score = np.mean(scores)
            logger.info(f"Params: {params}, Mean RMSLE: {mean_score:.4f}")
            
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
        
        logger.info(f"Best params: {best_params}, Best RMSLE: {best_score:.4f}")
        return best_params, best_score
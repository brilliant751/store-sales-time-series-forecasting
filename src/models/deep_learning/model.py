import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    tf_available = True
except ImportError:
    tf_available = False
    logging.warning("TensorFlow not installed, deep learning models will not be available")

logger = logging.getLogger(__name__)

class DeepLearningModel:
    def __init__(self, model_type='lstm', params=None):
        if not tf_available:
            raise ImportError("TensorFlow is required for deep learning models")
        
        self.model_type = model_type
        self.params = params or {
            'units': 128,
            'layers': 2,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'sequence_length': 28,
            'epochs': 50,
            'batch_size': 32
        }
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def create_model(self, n_features):
        model = Sequential()
        
        if self.model_type == 'lstm':
            layer_class = LSTM
        elif self.model_type == 'gru':
            layer_class = GRU
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")
        
        for i in range(self.params['layers']):
            if i == self.params['layers'] - 1:
                model.add(layer_class(
                    self.params['units'],
                    return_sequences=False,
                    input_shape=(self.params['sequence_length'], n_features) if i == 0 else None
                ))
            else:
                model.add(layer_class(
                    self.params['units'],
                    return_sequences=True,
                    input_shape=(self.params['sequence_length'], n_features) if i == 0 else None
                ))
            model.add(Dropout(self.params['dropout_rate']))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    
    def prepare_sequences(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        X_seq, y_seq = [], []
        seq_len = self.params['sequence_length']
        
        for i in range(seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i-seq_len:i])
            y_seq.append(y.iloc[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        n_features = X_train.shape[1]
        self.model = self.create_model(n_features)
        
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        
        callbacks = []
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=(X_val_seq, y_val_seq) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        X_scaled = self.scaler.transform(X)
        X_seq = []
        seq_len = self.params['sequence_length']
        
        for i in range(seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i-seq_len:i])
        
        if len(X_seq) == 0:
            return np.array([])
        
        predictions = self.model.predict(np.array(X_seq), verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        y_eval = y.iloc[self.params['sequence_length']:].values
        predictions = np.maximum(predictions, 0)
        
        rmsle = np.sqrt(mean_squared_log_error(y_eval, predictions))
        mae = mean_absolute_error(y_eval, predictions)
        rmse = np.sqrt(mean_squared_error(y_eval, predictions))
        
        return {
            'RMSLE': rmsle,
            'MAE': mae,
            'RMSE': rmse
        }
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        model_dir = '/'.join(filepath.split('/')[:-1])
        if model_dir:
            import os
            os.makedirs(model_dir, exist_ok=True)
        
        self.model.save(filepath)
        
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        pickle.dump(self.scaler, open(scaler_path, 'wb'))
        
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    def load_model(self, filepath):
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        
        logger.info(f"Model loaded from {filepath}")
        return self
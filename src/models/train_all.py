import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.xgboost.train import train_xgboost_model
from src.models.deep_learning.train import train_deep_learning_model
from src.evaluation.evaluator import ModelEvaluator

def main():
    print("Starting model training and evaluation pipeline...")
    
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    train_filepath = 'data/train.csv'
    
    if not os.path.exists(train_filepath):
        print(f"Error: Training data not found at {train_filepath}")
        print("Please place your train.csv in the data/ directory.")
        return
    
    evaluator = ModelEvaluator()
    
    print("\n=== Training XGBoost Model ===")
    try:
        xgb_model, xgb_train_metrics, xgb_val_metrics = train_xgboost_model(
            train_filepath,
            'outputs/models/xgboost_model.pkl',
            'logs/xgboost_training.log'
        )
        evaluator.add_metrics('XGBoost', xgb_val_metrics)
        print(f"XGBoost Validation Metrics: {xgb_val_metrics}")
    except Exception as e:
        print(f"Error training XGBoost: {e}")
    
    print("\n=== Training LSTM Model ===")
    try:
        lstm_model, lstm_train_metrics, lstm_val_metrics, _ = train_deep_learning_model(
            'lstm',
            train_filepath,
            'outputs/models/lstm_model.h5',
            'logs/lstm_training.log'
        )
        evaluator.add_metrics('LSTM', lstm_val_metrics)
        print(f"LSTM Validation Metrics: {lstm_val_metrics}")
    except Exception as e:
        print(f"Error training LSTM: {e}")
    
    print("\n=== Training GRU Model ===")
    try:
        gru_model, gru_train_metrics, gru_val_metrics, _ = train_deep_learning_model(
            'gru',
            train_filepath,
            'outputs/models/gru_model.h5',
            'logs/gru_training.log'
        )
        evaluator.add_metrics('GRU', gru_val_metrics)
        print(f"GRU Validation Metrics: {gru_val_metrics}")
    except Exception as e:
        print(f"Error training GRU: {e}")
    
    print("\n=== Generating Comparison Report ===")
    evaluator.plot_comparison()
    report = evaluator.generate_report()
    print("Comparison report generated.")
    
    best_model, best_metrics = evaluator.get_best_model()
    print(f"\nBest Model: {best_model}")
    print(f"Best Metrics: {best_metrics}")
    
    print("\nPipeline completed successfully!")

if __name__ == '__main__':
    main()
VENV := .venv
PIP_FLAGS ?=
ifeq ($(OS),Windows_NT)
PYTHON ?= py -3
VENV_PYTHON := $(VENV)/Scripts/python.exe
else
PYTHON ?= python3
VENV_PYTHON := $(VENV)/bin/python
endif

.PHONY: setup prepare-dirs train train-all train-xgboost train-lstm train-gru predict predict-all predict-xgboost predict-lstm predict-gru all

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install $(PIP_FLAGS) -r requirements.txt

prepare-dirs:
	$(VENV_PYTHON) -c "import os; [os.makedirs(p, exist_ok=True) for p in ['outputs/models', 'outputs/predictions', 'outputs/figures', 'logs']]"

train: train-all

train-all: prepare-dirs
	$(VENV_PYTHON) src/models/train_all.py

train-xgboost: prepare-dirs
	$(VENV_PYTHON) src/models/xgboost/train.py

train-lstm: prepare-dirs
	$(VENV_PYTHON) -c "from src.models.deep_learning.train import train_deep_learning_model; train_deep_learning_model('lstm', 'data/train.csv', 'outputs/models/lstm_model.h5', 'logs/lstm_training.log')"

train-gru: prepare-dirs
	$(VENV_PYTHON) -c "from src.models.deep_learning.train import train_deep_learning_model; train_deep_learning_model('gru', 'data/train.csv', 'outputs/models/gru_model.h5', 'logs/gru_training.log')"

predict: predict-all

predict-all: prepare-dirs
	$(VENV_PYTHON) src/models/xgboost/predict.py
	$(VENV_PYTHON) -c "from src.models.deep_learning.predict import predict_deep_learning_model; predict_deep_learning_model('lstm', 'outputs/models/lstm_model.h5', 'data/test.csv', 'outputs/predictions/lstm_predictions.csv')"
	$(VENV_PYTHON) -c "from src.models.deep_learning.predict import predict_deep_learning_model; predict_deep_learning_model('gru', 'outputs/models/gru_model.h5', 'data/test.csv', 'outputs/predictions/gru_predictions.csv')"

predict-xgboost: prepare-dirs
	$(VENV_PYTHON) src/models/xgboost/predict.py

predict-lstm: prepare-dirs
	$(VENV_PYTHON) -c "from src.models.deep_learning.predict import predict_deep_learning_model; predict_deep_learning_model('lstm', 'outputs/models/lstm_model.h5', 'data/test.csv', 'outputs/predictions/lstm_predictions.csv')"

predict-gru: prepare-dirs
	$(VENV_PYTHON) -c "from src.models.deep_learning.predict import predict_deep_learning_model; predict_deep_learning_model('gru', 'outputs/models/gru_model.h5', 'data/test.csv', 'outputs/predictions/gru_predictions.csv')"

all: train-all predict-all

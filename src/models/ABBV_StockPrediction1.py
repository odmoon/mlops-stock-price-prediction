import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import hydra
from omegaconf import DictConfig
from hydra import initialize, compose

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.logging.config import logger

print("Current working directory:", os.getcwd())
print("Absolute path of config directory:", os.path.abspath("/Users/dylanneal/Documents/mlops-stock-price-prediction/src/conf"))

class StockPredictor:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_data(self):
        data_path = hydra.utils.to_absolute_path(self.cfg.dataset.file_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return pd.read_csv(data_path)

    def preprocess_data(self, data):
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        features = data[['open', 'high', 'low', 'close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(features), scaler

    def create_dataset(self, data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back - 1):
            X.append(data[i:(i + look_back), :])
            Y.append(data[i + look_back, 3])  # We are predicting the 'close' price
        return np.array(X), np.array(Y)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(self.cfg.model.hidden_dim, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(self.cfg.model.dropout))
        model.add(LSTM(self.cfg.model.hidden_dim, return_sequences=False))
        model.add(Dropout(self.cfg.model.dropout))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.cfg.model.parameters.learning_rate),
                      loss='mean_squared_error')
        return model

    def train_model(self, model, trainX, trainY, testX, testY):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.cfg.training.early_stopping_patience)
        history = model.fit(trainX, trainY,
                            epochs=self.cfg.model.parameters.epochs,
                            batch_size=self.cfg.model.parameters.batch_size,
                            validation_data=(testX, testY),
                            callbacks=[early_stopping])
        return history

    def evaluate(self, model, trainX, trainY, testX, testY, scaler):
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        trainPredict = scaler.inverse_transform(np.concatenate((trainPredict, trainX[:, :, 1:]), axis=-1))[:, 0]
        testPredict = scaler.inverse_transform(np.concatenate((testPredict, testX[:, :, 1:]), axis=-1))[:, 0]
        trainY_original = scaler.inverse_transform(np.concatenate((trainY.reshape(-1, 1), trainX[:, :, 1]), axis=-1))[:, 0]
        testY_original = scaler.inverse_transform(np.concatenate((testY.reshape(-1, 1), testX[:, :, 1]), axis=-1))[:, 0]
        return trainY_original, trainPredict, testY_original, testPredict

    def plot_results(self, trainY_original, trainPredict, testY_original, testPredict, dates):
        plt.figure(figsize=(15, 8))
        plt.plot(dates[:len(trainY_original)], trainY_original, label='Train True')
        plt.plot(dates[:len(trainPredict)], trainPredict, label='Train Predict')
        plt.plot(dates[len(trainY_original):], testY_original, label='Test True')
        plt.plot(dates[len(trainPredict):], testPredict, label='Test Predict')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time (Year)')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()


@hydra.main(config_path="/Users/dylanneal/Documents/mlops-stock-price-prediction/src/conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    print("Running stock prediction with configuration:", cfg)
    
    wandb.init(project=cfg.wandb.project)
    wandb.config.update(cfg)

    stock_predictor = StockPredictor(cfg)

    data = stock_predictor.load_data()
    data_scaled, scaler = stock_predictor.preprocess_data(data)

    look_back = cfg.training.look_back
    X, Y = stock_predictor.create_dataset(data_scaled, look_back)

    train_size = int(len(X) * 0.8)
    trainX, testX = X[:train_size], X[train_size:]
    trainY, testY = Y[:train_size], Y[train_size:]

    model = stock_predictor.build_model((look_back, X.shape[2]))
    stock_predictor.train_model(model, trainX, trainY, testX, testY)

    trainY_original, trainPredict, testY_original, testPredict = stock_predictor.evaluate(model, trainX, trainY, testX, testY, scaler)

    dates = data.index[look_back+1:look_back+1+len(trainY_original)+len(testY_original)]
    stock_predictor.plot_results(trainY_original, trainPredict, testY_original, testPredict, dates)


if __name__ == "__main__":
    # Print the absolute path of the config file for debugging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.abspath(os.path.join(script_dir, "../conf"))
    config_file = os.path.join(config_dir, "config.yml")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Absolute path of config directory: {config_dir}")
    print(f"Absolute path of config.yml: {config_file}")
    
    # Check if config.yml exists
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
    else:
        print(f"Configuration file found: {config_file}")

    main()

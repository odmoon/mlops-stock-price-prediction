import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import wandb
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.logging.config import logger
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize Weights and Biases
    wandb.init(project='MLOPS-STOCK-PRICE-PREDICTION')
    config = wandb.config
    config.learning_rate = cfg.model.parameters.learning_rate

    logger.info("Weights and Biases initialized")

    # Set up the relative path to the file
    script_dir = hydra.utils.get_original_cwd()
    full_path = os.path.join(script_dir, cfg.dataset.path)

    # * Data is in the form of a CSV file * 

    # Load the dataset
    logger.info("Loading dataset")
    abbv_data = pd.read_csv(full_path)

    # Check if the data file exists
    if not os.path.exists(full_path):
        logger.error("Data file not found: %s", full_path)
        sys.exit("Data file not found")

    # * The 'date' column is converted to datetime format and set as the DataFrame index *
    abbv_data['date'] = pd.to_datetime(abbv_data['date'])
    abbv_data.set_index('date', inplace=True)

    # * Feature Selection for 'open, 'high', 'low', and 'close' prices for model input *
    features = abbv_data[['open', 'high', 'low', 'close']].values

    # * Feature scaling between 0 and 1 to improve neural network performance *
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(features)

    # * Look back defines how many previous timesteps are used to predict the next time step *
    look_back = cfg.training.look_back

    # Function to create dataset with multiple features
    def create_dataset(dataset, look_back=60):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), :]
            X.append(a)
            Y.append(dataset[i + look_back, -1])  # Target is still 'close' price
        return np.array(X), np.array(Y)

    logger.info("Creating dataset")
    X, y = create_dataset(data_scaled, look_back)

    # Splitting data into train and test sets
    train_size = int(len(X) * 0.67)
    #test_size = len(X) - train_size
    trainX, testX = X[:train_size], X[train_size:]
    trainY, testY = y[:train_size], y[train_size:]

    logger.info("Data split into training and testing sets")

    # Model Architecture
    logger.info("Creating and compiling the LSTM model")
    model = Sequential([
        Input(shape=(trainX.shape[1], 4)),
        LSTM(cfg.model.hidden_dim, return_sequences=True),
        Dropout(cfg.model.dropout),
        LSTM(cfg.model.hidden_dim),
        Dropout(cfg.model.dropout),
        Dense(1)
    ])

    adam_optimizer = Adam(learning_rate=cfg.model.parameters.learning_rate)
    model.compile(loss='mean_squared_error', optimizer=adam_optimizer)

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=cfg.training.early_stopping_patience, restore_best_weights=True)

    # Training the model
    logger.info("Training the model")
    history = model.fit(trainX, trainY, validation_split=cfg.training.validation_split, epochs=cfg.model.parameters.epochs, batch_size=cfg.model.parameters.batch_size, verbose=2, callbacks=[early_stopping])

    # Logging loss values to wandb
    for i in range(len(history.history['loss'])):
        wandb.log({"epoch": i + 1, "loss": history.history['loss'][i], "val_loss": history.history['val_loss'][i]})
        logger.info("Epoch %d: loss=%f, val_loss=%f", i + 1, history.history['loss'][i], history.history['val_loss'][i])

    # Making predictions
    logger.info("Making predictions")
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Inverting predictions to revert back to the original scale
    trainPredict = scaler.inverse_transform(np.c_[trainPredict, np.zeros((len(trainPredict), 3))])[:, 0]
    testPredict = scaler.inverse_transform(np.c_[testPredict, np.zeros((len(testPredict), 3))])[:, 0]
    trainY_original = scaler.inverse_transform(np.c_[trainY, np.zeros((len(trainY), 3))])[:, 0]
    testY_original = scaler.inverse_transform(np.c_[testY, np.zeros((len(testY), 3))])[:, 0]

    # Calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY_original, trainPredict))
    testScore = np.sqrt(mean_squared_error(testY_original, testPredict))
    print('Train Score: %.2f RMSE' % trainScore)
    print('Test Score: %.2f RMSE' % testScore)
    wandb.log({"train_rmse": trainScore, "test_rmse": testScore})

    # Extract date index for plotting
    dates = abbv_data.index[look_back+1:look_back+1+len(trainY_original)+len(testY_original)]

    # Plotting baseline and predictions
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:len(trainY_original)], trainY_original, label='Original Train')
    plt.plot(dates[:len(trainY_original)], trainPredict, label='Predicted Train')
    plt.plot(dates[len(trainY_original):], testY_original, label='Original Test')
    plt.plot(dates[len(trainY_original):], testPredict, label='Predicted Test')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time (Year)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate date labels for better visibility

    # Save the plot to the reports directory
    reports_dir = os.path.abspath('reports')
    os.makedirs(reports_dir, exist_ok=True)
    plot_path = os.path.join(reports_dir, 'stock_price_prediction.png')
    plt.savefig(plot_path)
    logger.info("Plot saved to %s", plot_path)

    plt.show()

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# * Data is in the form of a CSV file * 


# Load the dataset
data_path = '/Users/dylanneal/Documents/Assignment1/mlops-stock-price-prediction/src/data/stock_data/ABBV.csv'
abbv_data = pd.read_csv(data_path)


# * The 'date' column is converted to datetime format and set as the DataFrame index *

# Convert the 'date' column to datetime format and set as index
abbv_data['date'] = pd.to_datetime(abbv_data['date'])
abbv_data.set_index('date', inplace=True)



# * Feature Selection for 'open, 'high', 'low', and 'close' prices for model input *

# Select features to be used in the model
features = abbv_data[['open', 'high', 'low', 'close']].values


# * Feature scaling between 0 and 1 to improve neural network performance *

# Scale the features to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(features)


# * Look back defines how many previous timesteps are used to predict the next time step *
# Define look_back period
look_back = 60

# This function constructs input and output datasets for the LSTM.
# For each instance, it takes 'look_back' days of features as input
# and the next day's closing price as output

# Function to create dataset with multiple features
def create_dataset(dataset, look_back=60):
    
    X, Y = [], []
    
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, -1])  # Target is still 'close' price
        
    return np.array(X), np.array(Y)


# Create dataset
X, y = create_dataset(data_scaled)


# The dataset is split into training and testing parts based on a percentage of 67%

# Splitting data into train and test sets
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
trainX, testX = X[:train_size], X[train_size:]
trainY, testY = y[:train_size], y[train_size:]



# Model Architecture: The model consists of two LSTM layers
# interspersed with Dropoutlayers to prevent overfitting,
# and a Dense layer for output.

# The model is compiled with the mean squared error loss function and the adam optimizer.

# Create and compile the LSTM model
model = Sequential([
    Input(shape=(trainX.shape[1], 4)),
    
    LSTM(50, return_sequences=True),
    
    Dropout(0.2),
    
    LSTM(50),
    
    Dropout(0.2),
    
    Dense(1)
])

adam_optimizer = Adam(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer=adam_optimizer)


# Early Stopping is used to halt training when the validation loss hasn't improved for a
# specified number of epochs. This helps prevent overfitting.


# Setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# LET'S TRAIN

# The model is trained using the training data with
# validation split to monitor performance during training.


# Fit the model
history = model.fit(trainX, trainY, validation_split=0.2, epochs=100, batch_size=10, verbose=2, callbacks=[early_stopping])


# * Predictions are made for both training and testing datasets *

# Making predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# * Since the output was scaled, the predictions are rescaled back to the original 
# scale to interpret them in the context of actual stock prices. * 


# Inverting predictions to revert back to the original scale
trainPredict = scaler.inverse_transform(np.c_[trainPredict, np.zeros((len(trainPredict), 3))])[:, 0]
testPredict = scaler.inverse_transform(np.c_[testPredict, np.zeros((len(testPredict), 3))])[:, 0]
trainY_original = scaler.inverse_transform(np.c_[trainY, np.zeros((len(trainY), 3))])[:, 0]
testY_original = scaler.inverse_transform(np.c_[testY, np.zeros((len(testY), 3))])[:, 0]


# The Root Mean Squared Error (RMSE) is calculated for both training and testing
# predictions to evaluate the performance of the model.

# Calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY_original, trainPredict))
testScore = np.sqrt(mean_squared_error(testY_original, testPredict))
print('Train Score: %.2f RMSE' % trainScore)
print('Test Score: %.2f RMSE' % testScore)


# Extract date index for plotting
dates = abbv_data.index[look_back+1:look_back+1+len(trainY_original)+len(testY_original)]

# The original and predicted prices are plotted against time for both training and test datasets.

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
plt.show()




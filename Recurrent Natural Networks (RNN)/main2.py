import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_ds = pd.read_csv("Google_Stock_Price_Train.csv")
train_ds["Volume"] = train_ds["Volume"].str.replace(",", "")
training_set = train_ds.iloc[:, [1, 5]].values # only opening values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
timesteps = 60 # RNN will remember 60 data and will output 1 data

for i in range(timesteps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timesteps:i, 0:2])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))

# Building the RNN (Stacked LSTM)
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()

# Adding neurons
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 2))) # input layer
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=50)) # return_sequences=False
regressor.add(Dropout(rate=0.2))

regressor.add(Dense(units=1)) # output layer

# Compiling RNN
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Fitting RNN
regressor.fit(x=X_train, y=y_train, epochs=100, batch_size=32)

# Making the predictions and visualising the results
test_ds = pd.read_csv("Google_Stock_Price_Test.csv")
test_ds["Volume"] = test_ds["Volume"].str.replace(",", "")
test_set = test_ds.iloc[:, [1, 5]].values


# train_ds nin 1 ve 5. indexindeki verileri alıp bir dataframe oluşturuyoruz
dataset_total = pd.concat((train_ds.iloc[:, [1, 5]], test_ds.iloc[:, [1, 5]]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_ds) - timesteps:].values
inputs = sc.transform(inputs)

X_test = []

for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i-timesteps:i, 0:2])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))

predictions = regressor.predict(X_test)
predictions = np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1)
predictions = sc.inverse_transform(predictions)

# Visualising the results
# Visualising the results
plt.plot(test_set[:, 0], color='red', label='Real Google Stock Price (Open and Volume)')
plt.plot(predictions[:, 0], color='blue', label='Predicted Google Stock Price (Open and Volume)')
plt.title('Google Stock Price Prediction (Open and Volume)')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Export the model as keras file
regressor.save("rnn2.keras")

# Export the model as tensorflow file
import tensorflow as tf
from tensorflow import keras

model: Sequential = keras.models.load_model("rnn2.keras")

# Make predictions
predictions = regressor.predict(X_test)
predictions = np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1)
predictions = sc.inverse_transform(predictions)

# Visualising the results
plt.plot(test_set[:, 0], color='red', label='Real Google Stock Price (Open and Volume)')
plt.plot(predictions[:, 0], color='blue', label='Predicted Google Stock Price (Open and Volume)')
plt.title('Google Stock Price Prediction (Open and Volume)')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
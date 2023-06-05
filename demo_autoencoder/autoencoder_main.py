from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Somewhat on:
# https://keras.io/examples/timeseries/timeseries_anomaly_detection/

# SINGLE DIMENSIONAL INPUT
# x_train_raw = np.genfromtxt("upsampled_kills.csv", delimiter=",")
# x_test = x_train_raw.copy()
# duration = len(x_train_raw)  # this may need to change for multiple channels
# dims = 1

# MULTI DIMENSIONAL INPUT
x_train_raw1 = np.genfromtxt("upsampled_kills.csv", delimiter=",")
x_test1 = x_train_raw1.copy()
duration1 = len(x_train_raw1)  # this may need to change for multiple channels
dims = 2


# Normalize and save the mean and std we get,
# for normalizing test data.
training_mean = np.mean(x_train_raw)
training_std = np.std(x_train_raw)
training_value = (x_train_raw - training_mean) / training_std
print("Number of training samples:", len(training_value))
TIME_STEPS = 360  # 1 min of data

# Just adding some samples to complete the last minute.
samples_missing = 360 * ((duration // 360) + 1) - duration
training_value = np.append(training_value, np.zeros(samples_missing))


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(training_value)
print("Training input shape: ", x_train.shape)

model = keras.Sequential(
    [
        layers.Input(shape=(TIME_STEPS, dims)),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(np.squeeze(x_train_pred) - x_train))

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)

test_value = (x_train_raw - training_mean) / training_std
fig, ax = plt.subplots()

# Create sequences from test values.
x_test = create_sequences(test_value)
print("Test input shape: ", x_test.shape)

# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(np.squeeze(x_test_pred) - x_test))
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))


anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)


df_subset = pd.DataFrame(test_value).iloc[anomalous_data_indices]
fig, ax = plt.subplots()
pd.DataFrame(test_value).plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()

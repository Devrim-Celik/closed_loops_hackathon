import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data import data_pair

sns.set(style='whitegrid', palette='muted', font_scale=1.5)


RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

X, y = data_pair()

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
trainX, testX = X.iloc[0:train_size], X.iloc[train_size:len(X)]
trainY, testY = y.iloc[0:train_size], y.iloc[train_size:len(X)]
print(len(trainX), len(testX))


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 10

# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(trainX, trainY, time_steps)
X_test, y_test = create_dataset(testX, testY, time_steps)

print(X_train.shape, y_train.shape)

model = keras.Sequential()
model.add(keras.layers.LSTM(
  units=265,
  input_shape=([X_train.shape[1], X_train.shape[2]])
))
model.add(keras.layers.Dense(units=1))
model.compile(
  loss='mean_squared_error',
  optimizer=keras.optimizers.Adam(0.0005)
)
print(model.summary())
print("\n\n")
history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    shuffle=False
)

y_pred = model.predict(X_test)

plt.figure()
plt.plot(y_train)
plt.plot(y_pred)
#plt.plot(X_train)
plt.show()

print(y_pred.shape)
print(type(history))
print(history.keys())

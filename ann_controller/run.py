import os
import sys
import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

from config import Config
from lstm import Model
from data import load_batch, data_pair
from ode import ODE

PRINT_STEP = 1
TEST_TIME = 250
print("REAL VERSION")


def mse(u, v):
    return np.sum(np.square(u-v))/u.shape[0]

if __name__ == '__main__':
    # config
    config = Config()

    # model with config
    model = Model(config)

    error = []

    # to keep it simple, we define the number of beatches we wanna do
    for batch_indx in range(config.batch_number):
        # load random snippets from one file
        x, t = load_batch(config)#
        # for each of them generate an initial state
        initial_state = np.zeros((config.batch_number, 6))
        with tf.GradientTape() as tape:
            output = model(x)
            train_loss = config.loss_function(t, output)
            train_error = mse(t, output)
            gradients = tape.gradient(train_loss, model.trainable_variables)

        if batch_indx % PRINT_STEP == 0:
            print("* Batch: [{:4d}/{}] === Loss: {:04.5f} === Error: {:04.5f}".format(batch_indx,
                                    config.batch_number,
                                    train_loss,
                                    train_error))

        config.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        error.append(train_error)

        if batch_indx % TEST_TIME == 0 and batch_indx != 0:

            plt.figure()
            plt.plot(error)

            Zh, I = data_pair()
            Zh = np.expand_dims(np.array(Zh),0)
            I = np.array(I)
            I_pred = model(Zh)

            print("STARTING SAVE", I.shape)
            df = pd.DataFrame()
            df["optimal_current"] = I_pred
            df.to_save("./optimal"+str(batch_indx)+".csv")
            print("FINISHED SAVE")
            plt.figure()
            plt.plot(I[:,0], color="blue", label="optimal")
            plt.plot(I_pred[0, :, 0], color="red", label="prediction")
            plt.plot(Zh[0,:,0], color="black", label="profile")
            plt.legend()
            plt.show()

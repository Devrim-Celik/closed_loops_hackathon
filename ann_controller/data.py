import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Config

Zh_DATA_DIR = "/home/student/d/dcelik/Desktop/street"
I_DATA_DIR = "/home/student/d/dcelik/Desktop/data"

def data_pair():
    ZH_files = os.listdir(Zh_DATA_DIR)
    I_files = os.listdir(I_DATA_DIR)
    # TODO dont get just one...
    file = np.random.choice(ZH_files)
    I_file = [I for I in I_files if file[:-4] in I][0]

    I = pd.read_csv(I_DATA_DIR + "/" + I_file , names=["I"])

    Zh = pd.read_csv(Zh_DATA_DIR + "/" + file, names=["profile"])

    """
    dt = 0.005

    time = [dt*indx for indx in range(len(I))]
    """
    if len(Zh) != len(I):
        raise Exception("Zh and I do not have same length!")
    return Zh, I


def load_batch(config):

    Zh, I = data_pair()

    start_points = np.random.randint(len(Zh)-config.time_steps, size=config.batch_size)

    X = np.zeros((config.batch_size, config.time_steps, 1))
    y = np.zeros((config.batch_size, config.time_steps, 1))

    for batch in range(config.batch_size):
            X[batch, :] = Zh[start_points[batch]:start_points[batch]+config.time_steps]
            y[batch, :] = I[start_points[batch]:start_points[batch]+config.time_steps]

            batch_norm_mean = np.sum(X[batch, :], axis=0) / X[batch, :].shape[0]
            batch_norm_var = np.var(X[batch, :], axis=0) / X[batch, :].shape[0]
            #print("Before:", batch_norm_mean, batch_norm_mean)
            X = (X - batch_norm_mean) / (batch_norm_var + 1e-8) ** 0.5
            #batch_norm_mean = np.sum(X[batch, :], axis=0) / X[batch, :].shape[0]
            #batch_norm_var = np.var(X[batch, :], axis=0) / X[batch, :].shape[0]
            #print("After:", batch_norm_mean, batch_norm_mean)
    # batchnorm


    return X, y

if __name__=="__main__":
    c = Config()
    X, y = load_batch(c)
    print(X.shape, y.shape)

    for i in range(16):
        plt.figure()
        plt.plot(X[i])
        plt.show()

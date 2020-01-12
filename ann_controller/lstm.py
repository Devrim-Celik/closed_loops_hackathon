import numpy as np
import tensorflow as tf

from ode import ODE

from tensorflow.keras.layers import Layer, Dense, LayerNormalization
from tensorflow.keras.activations import softmax, relu, tanh, sigmoid
from tensorflow.python.keras.layers.recurrent import RNN, SimpleRNN, LSTM, GRU
from tensorflow.python.keras import activations, initializers
from tensorflow.python.keras import backend as K
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape

tf.keras.backend.set_floatx('float64')

class Model(Layer):
    """
    Model for applying recurrent layers with softmax
    """
    def __init__(self, config, **kwargs):
        """
        Init

        Args
            config
        """
        super(Model, self).__init__(**kwargs)

        self.lstm1 = LSTM(config.units_layer1, return_sequences=True)
        self.lstm2 = LSTM(config.units_layer2)
        self.lstm3 = LSTM(config.units_layer3)
        self.lstm4 = LSTM(config.units_layer4)


        self.LN = LayerNormalization(epsilon=1e-6)
        self.dense = Dense(1, activation=sigmoid)

    def call(self, x):
        '''
        Call method for performing sequential calculation of input.

        Args:
            x (np.array):  input of shape (batch_size, time_step, 1+8)
        returns
            output
        '''
        o1 = self.LN(self.lstm1(x))
        """
        o1 = tf.expand_dims(o1, 2)
        print(o1.shape)
        o2 = self.LN(self.lstm2(o1))
        o1 = tf.expand_dims(o2, 2)
        o3 = self.LN(self.lstm3(o2))
        o1 = tf.expand_dims(o3, 2)
        o4 = self.LN(self.lstm4(o3))
        o1 = tf.expand_dims(o4, 2)
        o5 = self.dense(o4)
        """
        o5 = 2*self.dense(o1)
        return o5

    def __repr__(self):
      return "{}({})".format(self.type, self.rnn_units)

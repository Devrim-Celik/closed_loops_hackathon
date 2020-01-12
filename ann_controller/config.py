from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

class Config(object):

    def __init__(self):
        self.batch_number = 1000
        self.batch_size = 16
        self.time_steps = 100

        self.units_layer1 = 128
        self.units_layer2 = 128
        self.units_layer3 = 32
        self.units_layer4 = 16

        self.tf_layer_norm = False

        self.learning_rate = 0.0001
        self.optimizer = Adam(self.learning_rate)
        self.loss_function = MeanSquaredError()
        self.metric = None


    def __repr__(self):
        """
        This function is called when your print(config). The idea is to print a full representation of all the
        settings that were used. I print this summary to a text file that I store in the checkpoint directory,
        which might come in handy if you are wondering which exact set of parameters was used for a certain run.
        """
        return "\n".join([str(key)+": "+str(value) for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)])

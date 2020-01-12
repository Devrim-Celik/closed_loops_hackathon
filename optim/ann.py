###############################################################################
#
# Functions for constructing and running a simple feed-forward ANN in Numpy
#
###############################################################################

import numpy as np

def create_init_params(input_shape, l1_size, l2_size, output_shape):
    '''
    Create inital parameters for the network.
    '''
    in_weights = np.random.uniform(
        low=-1., high=1., size=(input_shape, l1_size))
    l1_weights = np.random.uniform(
        low=-1., high=1., size=(l1_size, l2_size))
    l2_weights = np.random.uniform(
        low=-1., high=1., size=(l2_size, output_shape))
    return [in_weights, l1_weights, l2_weights]

def sigmoid(inpt):
    return 1. / (1. + np.exp(-1 * inpt))

def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result

def network_function(weights):
    def fun(x):
        x = relu(np.matmul(x, weights[0]))
        x = relu(np.matmul(x, weights[1]))
        x = sigmoid(np.matmul(x, weights[2]))
        x += 1 # scale to interval (0, 2)
        return x
    return fun

def weights_to_vec(arr):
    return np.concatenate((arr[0].flatten(), arr[1].flatten(), arr[2].flatten()))

def vec_to_mat(input_shape, l1_size, l2_size, output_shape, vec):
    indx1 = input_shape * l1_size
    indx2 = indx1 + l1_size*l2_size
    in_weights = vec[:indx1].reshape((input_shape, l1_size))
    l1_weights = vec[indx1:indx2].reshape((l1_size, l2_size))
    l2_weights = vec[indx2:].reshape((l2_size, output_shape))
    return [in_weights, l1_weights, l2_weights]

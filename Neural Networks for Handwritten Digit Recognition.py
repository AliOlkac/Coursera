# UNQ_C1
# GRADED CELL: Sequential model
import tensorflow as tf
import keras
from keras import layers
from keras import ops
from keras import Sequential
import numpy as np



model = Sequential(
    [
        tf.keras.Input(shape=(400,)),  # specify input size
        ### START CODE HERE ###
        layers.Dense(25,activation='sigmoid'),
        layers.Dense(5,activation='sigmoid'),
        layers.Dense(1,activation='sigmoid')

        ### END CODE HERE ###
    ], name="my_model"
)


# UNQ_C2
# GRADED FUNCTION: my_dense

def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    ### START CODE HERE ###
    for j in range(units):
        w = W[:,j]
        z = np.dot(a_in,w)+b[j]
        a_out[j] = g(z)
    ### END CODE HERE ###
    return (a_out)


# UNQ_C1
# GRADED CELL: my_softmax

def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    ### START CODE HERE ###
    z_exp = np.exp(z)
    z_sum = np.sum(z_exp)
    a = z_exp/z_sum

    ### END CODE HERE ###
    return a


# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234)
model = Sequential(
    [
        ### START CODE HERE ###
        tf.keras.Input(shape=(400,)),     # @REPLACE
        Dense(25, activation='relu', name = "L1"), # @REPLACE
        Dense(15, activation='relu',  name = "L2"), # @REPLACE
        Dense(10, activation='linear', name = "L3"),  # @REPLACE
        ### END CODE HERE ###
    ], name = "my_model"
)
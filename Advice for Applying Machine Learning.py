# UNQ_C1
# GRADED CELL: eval_mse
from keras import Sequential
from keras.src.layers import Dense
from tensorboard.compat import tf


def eval_mse(y, yhat):
    """
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)
    """
    m = len(y)
    err = 0.0
    for i in range(m):
    ### START CODE HERE ###
        err += (yhat[i]-y[i])**2
    err = err/2*m
    ### END CODE HERE ###

    return (err)


# UNQ_C2
# GRADED CELL: eval_cat_err
def eval_cat_err(y, yhat):
    """
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
    ### START CODE HERE ###
        if y[i] != yhat[i]:
            incorrect += 1
    cerr = incorrect/m
    ### END CODE HERE ###

    return (cerr)


# UNQ_C3
# GRADED CELL: model
import logging
"""Dense layer with 120 units, relu activation
Dense layer with 40 units, relu activation
Dense layer with 6 units and a linear activation (not softmax)
Compile using
loss with SparseCategoricalCrossentropy, remember to use from_logits=True
Adam optimizer with learning rate of 0.01."""
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(1234)
model = Sequential(
    [
        ### START CODE HERE ###
        Dense(120, activation='relu'),
        Dense(40, activation='relu'),
        Dense(6, activation='linear')
        ### END CODE HERE ###

    ], name="Complex"
)
model.compile(
    ### START CODE HERE ###
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    ### END CODE HERE ###
)

# UNQ_C4
# GRADED CELL: model_s

tf.random.set_seed(1234)
model_s = Sequential(
    [
        ### START CODE HERE ###
        Dense(6, activation='relu'),
        Dense(6, activation='linear'),
        ### END CODE HERE ###
    ], name="Simple"
)
model_s.compile(
    ### START CODE HERE ###
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
    ### START CODE HERE ###
)

# UNQ_C5
# GRADED CELL: model_r

tf.random.set_seed(1234)
model_r = Sequential(
    [
        ### START CODE HERE ###
        Dense(120,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(40,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(6,activation='linear')
        ### START CODE HERE ###
    ], name=None
)
model_r.compile(
    ### START CODE HERE ###
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    ### START CODE HERE ###
)

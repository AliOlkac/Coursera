import numpy as np


# GRADED FUNCTION: cofi_cost_func
# UNQ_C1

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user

      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    ### START CODE HERE ###
    for j in range(nu):
        w = W[j, :]
        b_j = b[0, j]
        for i in range(nm):
            x = X[i, :]
            y = Y[i, j]
            r = R[i, j]
            J += np.square(r * (np.dot(w, x) + b_j - y))
    J = J / 2
    J += lambda_ / 2 * (np.sum(np.square(W)) + np.sum(np.square(X)))
    ### END CODE HERE ###

    return J
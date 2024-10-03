"""
Gradient descent for linear regression exercises.
"""
import numpy as np


def predict(x, w):
    """Return h(x) = w^T x
    Parameters:
        x - d-dimensional numpy array (with bias value 1 at position 0)
        w - d-dimensional weight array
    """
    return -1


def calc_sse(X, w, y):
    """Return .5 * sse for the data in X

    Parameters:
        X - n x d numpy array, n points dimensionality d (with 1's in col 0)
        w - d-dimensional weight array
        y - length-n numpy array of target values
    """
    return -1

def batch_gd(X, w, y, eta):
    """Perform one round of batch gradient descent and return the new 
    weight vector.

    Parameters:
        X - n x d numpy array, n points dimensionality d (with 1's in col 0)
        w - d-dimensional weight array
        y - length-n numpy array of target values
        eta - learning rate
    """
    return -1
 
def stochastic_gd(X, w, y, eta):
    """Perform one round of stochastic gradient descent and return the new 
    weight vector.

    Parameters:
        X - n x d numpy array, n points dimensionality d (with 1's in col 0)
        w - d-dimensional weight array
        y - length-n numpy array of target values
        eta - learning rate
    """
    return -1


def main():
    """Print answers to the gradient descent activity."""

    # It is convenient to pre-prepend a column of 1's to facilitate
    # the bias weights.
    X = np.array([[1, 1, 2.],
                  [1, -2, 5],
                  [1, 0, 1]])
    y = np.array([1, 6, 1])
    w = np.array([0, 1, .5])

    print("1a:")


    print("\n1b:")


    print("\n2:")


    print("\n3:")


if __name__ == "__main__":
    main()

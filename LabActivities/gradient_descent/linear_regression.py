"""
name: Sayemum Hassan

Gradient descent for linear regression exercises.
"""
import numpy as np


def predict(x, w):
    """Return h(x) = w^T x
    Parameters:
        x - d-dimensional numpy array (with bias value 1 at position 0)
        w - d-dimensional weight array
    """
    return w.transpose() @ x


def calc_sse(X, w, y):
    """Return .5 * sse for the data in X

    Parameters:
        X - n x d numpy array, n points dimensionality d (with 1's in col 0)
        w - d-dimensional weight array
        y - length-n numpy array of target values
    """
    summed = 0
    for i in range(len(y)):
        predicted = predict(X[i], w)
        summed += y[i] - predicted
    
    sse = (1/2) * (summed ** 2)
    
    return .5 * sse

def batch_gd(X, w, y, eta):
    """Perform one round of batch gradient descent and return the new 
    weight vector.

    Parameters:
        X - n x d numpy array, n points dimensionality d (with 1's in col 0)
        w - d-dimensional weight array
        y - length-n numpy array of target values
        eta - learning rate
    """
    m = len(y)
    predictions = X @ w
    errors = predictions - y
    gradient = (X.T @ errors) / m
    new_w = w - eta * gradient
    
    return new_w
 
def stochastic_gd(X, w, y, eta):
    """Perform one round of stochastic gradient descent and return the new 
    weight vector.

    Parameters:
        X - n x d numpy array, n points dimensionality d (with 1's in col 0)
        w - d-dimensional weight array
        y - length-n numpy array of target values
        eta - learning rate
    """
    for i in range(len(y)):
        predicted = predict(X[i], w)
        error = predicted - y[i]
        gradient = error * X[i]
        w = w - eta * gradient
    
    return w


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
    print(predict(np.array([1, 2, 3]), w))

    print("\n1b:")
    print(calc_sse(X, w, y))

    print("\n2:")
    print(batch_gd(X, w, y, .01))

    print("\n3:")
    print(stochastic_gd(X, w, y, .01))

if __name__ == "__main__":
    main()

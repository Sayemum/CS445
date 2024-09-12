""" Artificial Data Source for KNN Experiments

author: Nathan Sprague
version: 2/5/20

"""

import numpy as np
import sklearn.datasets


def get_iris_data():
    np.random.seed(20)
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X_new = np.zeros((X.shape[0], 6))
    X_new[:, 0] = np.random.random((X.shape[0], )) * 15
    X_new[:, 1] = X[:, 0]
    X_new[:, 2] = X[:, 1] 
    X_new[:, 3] = np.random.normal(3, 10., (X.shape[0],))
    X_new[:, 4] = X[:, 2]
    X_new[:, 5] = X[:, 3]
    return X_new, y

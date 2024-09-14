"""Pure Python Decision Tree Classifier and Regressor.

Simple binary decision tree classifier and regressor.
Splits for classification are based on Gini impurity. Splits for
regression are based on variance.

Author: CS445 Instructor and Sayemum Hassan
Version: 1

"""
from collections import namedtuple, Counter
import numpy as np
from abc import ABC

# Named tuple is a quick way to create a simple wrapper class...
Split_ = namedtuple('Split',
                    ['dim', 'pos', 'X_left', 'y_left', 'counts_left',
                     'X_right', 'y_right', 'counts_right'])


class Split(Split_):
    """
    Represents a possible split point during the decision tree
    creation process.

    Attributes:

        dim (int): the dimension along which to split
        pos (float): the position of the split
        X_left (ndarray): all X entries that are <= to the split position
        y_left (ndarray): labels corresponding to X_left
        counts_left (Counter): label counts
        X_right (ndarray):  all X entries that are > the split position
        y_right (ndarray): labels corresponding to X_right
        counts_right (Counter): label counts
    """

    def __repr__(self):
        result = "Split(dim={}, pos={},\nX_left=\n".format(self.dim,
                                                           self.pos)
        result += repr(self.X_left) + ",\ny_left="
        result += repr(self.y_left) + ",\ncounts_left="
        result += repr(self.counts_left) + ",\nX_right=\n"
        result += repr(self.X_right) + ",\ny_right="
        result += repr(self.y_right) + ",\ncounts_right="
        result += repr(self.counts_right) + ")"

        return result


def split_generator(X, y, keep_counts=True):
    """
    Utility method for generating all possible splits of a data set
    for the decision tree construction algorithm.

    :param X: Numpy array with shape (num_samples, num_features)
    :param y: Numpy array with length num_samples
    :param keep_counts: Maintain counters (only useful for classification.)
    :return: A generator for Split objects that will yield all
            possible splits of the data
    """

    # Loop over all of the dimensions.
    for dim in range(X.shape[1]):
        if np.issubdtype(y.dtype, np.integer):
            counts_left = Counter()
            counts_right = Counter(y)
        else:
            counts_left = None
            counts_right = None

        # Get the indices in sorted order so we can sort both data and labels
        ind = np.argsort(X[:, dim])

        # Copy the data and the labels in sorted order
        X_sort = X[ind, :]
        y_sort = y[ind]

        last_split = 0
        # Loop through the midpoints between each point in the
        # current dimension
        for index in range(1, X_sort.shape[0]):

            # don't try to split between equal points.
            if X_sort[index - 1, dim] != X_sort[index, dim]:
                pos = (X_sort[index - 1, dim] + X_sort[index, dim]) / 2.0

                if np.issubdtype(y.dtype, np.integer):
                    flipped_counts = Counter(y_sort[last_split:index])
                    counts_left = counts_left + flipped_counts
                    counts_right = counts_right - flipped_counts

                last_split = index
                # Yield a possible split.  Note that the slicing here does
                # not make a copy, so this should be relatively fast.
                yield Split(dim, pos,
                            X_sort[0:index, :], y_sort[0:index], counts_left,
                            X_sort[index::, :], y_sort[index::], counts_right)


def impurity(y, y_counts=None):
    """ Calculate Gini impurity for the class labels y.
        If y_counts is provided it will be the counts of the labels in y.
    """
    if y_counts is None:
        # calculate count of every class in y
        y_counts = np.array(list(Counter(y).values()), dtype=float)
    else:
        # y_counts is an iterable so we convert to numpy array
        y_counts = np.array(list(y_counts.values()), dtype=float)

    # total samples
    total_samples = np.sum(y_counts)

    # proportions of each class
    proportions = y_counts / total_samples

    # return gini inpurity
    return 1 - np.sum(proportions ** 2)


def weighted_impurity(split):
    """ Weighted gini impurity for a possible split. """ 
    total_samples = len(split.y_left) + len(split.y_right)

    # calculate impurity of left and right splits
    impurity_left = impurity(split.y_left, split.counts_left)
    impurity_right = impurity(split.y_right, split.counts_right)

    # weighted impurity
    weighted_impurity = (len(split.y_left) / total_samples * impurity_left) + (len(split.y_right) / total_samples * impurity_right)

    return weighted_impurity


class DecisionTree(ABC):
    """
    A binary decision tree for use with real-valued attributes.

    """

    def __init__(self, max_depth=np.inf):
        """
        Decision tree constructor.

        :param max_depth: limit on the tree depth.
                          A depth 0 tree will have no splits.
        """
        self.max_depth = max_depth
        
        # initialize the root of tree
        self.root = None

    def fit(self, X, y):
        """
        Construct the decision tree using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy array with length num_samples
        """
        # LOGIC MIGHT BE THE SAME FOR BOTH TREE TYPES?
        raise NotImplementedError()

    def predict(self, X):
        """
        Predict labels for a data set by finding the appropriate leaf node for
        each input and using either the majority label or the mean value
        as the prediction.

        :param X:  Numpy array with shape (num_samples, num_features)
        :return: A length num_samples numpy array containing predictions.
        """
        # LOGIC CAN PROBABLY BE THE SAME FOR BOTH TREE TYPES?
        # (DEPENDS ON HOW YOU IMPLEMENT YOUR NODE CLASSES)
        raise NotImplementedError()

    def get_depth(self):
        """
        :return: The depth of the decision tree.
        """
        # LOGIC SHOULD BE THE SAME FOR BOTH TREE TYPES
        # raise NotImplementedError()
        return self.max_depth


class DecisionTreeClassifier(DecisionTree):
    """
    A binary decision tree classifier for use with real-valued attributes.

    """
    # pass
    # Feel free to add methods as needed. Avoid code duplication by keeping
    # common functionality in the superclass.
    def fit(self, X, y):
        # target val is categorical
        # split data based on gini impurity
        # predict class label for given input data
        pass
        
        # split dataset into testing and training
        


class DecisionTreeRegressor(DecisionTree):
    """
    A binary decision tree regressor for use with real-valued attributes.

    """
    # pass
    # Feel free to add methods as needed. Avoid code duplication by keeping
    # common functionality in the superclass.
    def fit(self, X, y):
        # target val is continuous
        # split data based on MSE or variance reduction
        # predict continuous val for given input data
        pass


class Node:
    """
    It will probably be useful to have a Node class.  In order to use the
    visualization code in draw_trees, the node class must have three
    attributes:

    Attributes:
        left:  A Node object or Null for leaves.
        right: A Node object or Null for leaves.
        split: A Split object representing the split at this node,
                or Null for leaves
    """
    # pass
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threashold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value


def tree_demo():
    """Simple illustration of creating and drawing a tree classifier."""
    import draw_tree
    X = np.array([[0.88, 0.39],
                  [0.49, 0.52],
                  [0.68, 0.26],
                  [0.57, 0.51],
                  [0.61, 0.73]])
    y = np.array([1, 0, 0, 0, 1])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    draw_tree.draw_tree(X, y, tree)


if __name__ == "__main__":
    tree_demo()

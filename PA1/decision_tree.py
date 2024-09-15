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
        self._root = None

    def fit(self, X, y):
        """
        Construct the decision tree using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy array with length num_samples
        """
        # LOGIC MIGHT BE THE SAME FOR BOTH TREE TYPES?
        self._root = self.build_tree(X, y, depth=0)

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
        predictions = [self.predict_row(row, self._root) for row in X]
        return np.array(predictions)

    def get_depth(self):
        """
        :return: The depth of the decision tree.
        """
        # LOGIC SHOULD BE THE SAME FOR BOTH TREE TYPES
        return self._get_depth(self._root)
    
    def _get_depth(self, node):
        """
        Recursively find the depth of the tree.

        :param node: Current node
        :return: Depth of the tree
        """
        if node is None or node.value is not None:
            return 0
        
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))

    def build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy array with length num_samples
        :param depth: Current depth of the tree
        :return: Root node of the tree
        """
        if depth >= self.max_depth or len(set(y)) == 1:
            return Node(value=self.leaf_value(y))

        split = self.best_split(X, y)
        if split is None:
            return Node(value=self.leaf_value(y))

        node = Node(split=split)
        node.left = self.build_tree(split.X_left, split.y_left, depth + 1)
        node.right = self.build_tree(split.X_right, split.y_right, depth + 1)
        
        return node

    def predict_row(self, row, node):
        """
        Predict the label for a single row of input.

        :param row: A single data point
        :param node: Current node
        :return: Predicted label or value
        """
        if node.value is not None:
            return node.value
        if row[node.split.dim] <= node.split.pos:
            return self.predict_row(row, node.left)
        
        return self.predict_row(row, node.right)


class DecisionTreeClassifier(DecisionTree):
    """
    A binary decision tree classifier for use with real-valued attributes.

    """
    # Feel free to add methods as needed. Avoid code duplication by keeping
    # common functionality in the superclass.
    def gini_impurity(self, y, y_counts=None):
        """
        Calculate the Gini impurity for a list of labels.

        :param y: Numpy array of labels y
        :return: The Gini impurity
        """
        counts = y_counts or Counter(y)
        impurity = 1.0
        
        for label in counts:
            probability = counts[label] / len(y)
            impurity -= probability ** 2
        
        return impurity

    def best_split(self, X, y):
        """
        Find the best split for the data.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy array with length num_samples
        :return: A Split object representing the best split, or None if no split is found
        """
        best_split = None
        best_impurity = float('inf')

        for split in split_generator(X, y):
            if split.y_left.size > 0 and split.y_right.size > 0:
                left_impurity = self.gini_impurity(split.y_left)
                right_impurity = self.gini_impurity(split.y_right)
                total_impurity = (len(split.y_left) * left_impurity +
                                  len(split.y_right) * right_impurity) / len(y)

                if total_impurity < best_impurity:
                    best_impurity = total_impurity
                    best_split = split

        return best_split

    def leaf_value(self, y):
        """
        Determine the value for a leaf node.

        :param y: Numpy array of labels
        :return: The value for the leaf node
        """
        return Counter(y).most_common(1)[0][0]


class DecisionTreeRegressor(DecisionTree):
    """
    A binary decision tree regressor for use with real-valued attributes.

    """
    # Feel free to add methods as needed. Avoid code duplication by keeping
    # common functionality in the superclass.
    def variance(self, y):
        """
        Calculate the variance for a list of values.

        :param y: Numpy array of values
        :return: The variance
        """
        return np.var(y)

    def best_split(self, X, y):
        """
        Find the best split for the data.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy array with length num_samples
        :return: A Split object representing the best split, or None if no split is found
        """
        best_split = None
        best_variance = float('inf')

        for split in split_generator(X, y):
            if split.y_left.size > 0 and split.y_right.size > 0:
                left_variance = self.variance(split.y_left)
                right_variance = self.variance(split.y_right)
                total_variance = (len(split.y_left) * left_variance +
                                  len(split.y_right) * right_variance) / len(y)

                if total_variance < best_variance:
                    best_variance = total_variance
                    best_split = split

        return best_split

    def leaf_value(self, y):
        """
        Determine the value for a leaf node.

        :param y: Numpy array of values
        :return: The value for the leaf node
        """
        return np.mean(y)


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
    def __init__(self, left=None, right=None, split=None, value=None):
        self.left = left
        self.right = right
        self.split = split
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
    # tree = DecisionTreeClassifier()
    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    draw_tree.draw_tree(X, y, tree)


if __name__ == "__main__":
    tree_demo()

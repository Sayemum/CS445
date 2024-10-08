"""Pure Python Naive Bayes classifier

Simple nb_classifier.

Initial Author: Kevin Molloy and Sayemum Hassan
Minor modifications by: Nathan Sprague
"""

import numpy as np


class NBClassifier:
    """A naive bayes classifier for use with categorical and real-valued attributes/features.

    Public Attributes:
        classes (list): The set of integer classes this tree can classify.
        smoothing_flag (boolean): Indicator whether or not to perform
                                  Laplace smoothing
        feature_dists (list):  A placeholder for each feature/column in X
                               that holds the class-conditional distributions
                               for that feature. The distributions are
                               represented as dictionaries that map from
                               class labels to:
                                -- for continuous features, a tuple with
                                the distribution parameters for a Gaussian
                                (mean, variance)
                                -- for discrete features, another dictionary
                                where the keys are the individual domain
                                values for the feature,and the value is the
                                computed probability from the training data
        priors (dictionary): A dictionary mapping from class labels to
                             probabilities.

    """

    def __init__(self, smoothing_flag=False):
        """
        NBClassifier constructor.

        :param smoothing_flag: for discrete attributes only
        """
        self.smoothing_flag = smoothing_flag
        self.feature_dists = []
        self.priors = None
        self.classes = None

    def fit(self, X, X_categorical, y):
        """
        Construct the NB using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features).
                  This is the training data.
        :param X_categorical: numpy boolean array with length num_features.
                              True values indicate that the feature is discrete.
                              False values indicate that the feature is continuous.
        :param y: Numpy integer array with length num_samples
                  These are the training labels.

        :return: Stores results in class variables, nothing returned.

        An example of how my dictionary looked after running fit on the
        loan classification problem in the textbook without smoothing:
        [{0: {'No': 0.5714285714285714, 'Yes': 0.42857142857142855},
         1: {'No': 1.0}   },
        {0: {'Divorced': 0.14285714285714285, 'Married': 0.57, 'Single': 0.28},
         1: {'Divorced': 0.3333333333333333, 'Single': 0.6666666666666666}   },
        {0: (110.0, 54.543560573178574),
         1: (90.0, 5.0)}]
        """

        # Need a category for each column in X
        assert (X.shape[1] == X_categorical.shape[0])

        # each row in training data needs a label
        assert (X.shape[0] == y.shape[0])
        
        # calculate prior probabilities
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.priors = {item: np.sum(y == item) / n_samples for item in self.classes}
        
        # class conditional distributions
        for feature_index in range(n_features):
            feature_dist = {}
            if X_categorical[feature_index]:
                # categorical features: comput probability
                all_values = np.unique(X[:, feature_index])
                
                for class_item in self.classes:
                    class_mask = (y == class_item)
                    values, counts = np.unique(X[class_mask, feature_index], return_counts=True)
                    total_counts = np.sum(counts)
                    
                    smooth_counts = counts  # if smoothing_flag is false
                    if self.smoothing_flag:
                        # apply laplace smoothing
                        smooth_counts = {val: 1 for val in all_values}
                        for i, val in enumerate(values):
                            smooth_counts[val] += counts[i]
                            
                        total_counts += len(all_values)
                    else:
                        smooth_counts = {val: 0 for val in all_values}
                        for i, val in enumerate(values):
                            smooth_counts[val] = counts[i]
                    
                    feature_dist[class_item] = \
                        {val: smooth_counts[val] / total_counts for val in all_values}
            else:
                # continuous features: compute mean and variance for each class
                for class_item in self.classes:
                    class_mask = (y == class_item)
                    feature_values = X[class_mask, feature_index].astype(float)
                    mean = np.mean(feature_values)
                    variance = np.var(feature_values, ddof=1) + 1e-9
                    feature_dist[class_item] = (mean, variance)
                
            self.feature_dists.append(feature_dist)

    def feature_class_prob(self, feature_index, x, class_label):
        """
        Compute a single class-conditional probability.  You can call
        this function in your predict function if you wish.

        Example: For the loan default problem:
            feature_class_prob(1, 'Single', 0) returns 0.2857

        :param feature_index:  index into the feature set (column of X)
        :param x: the data value
        :param class_label: the label used in the probability (see return below)

        :return: p(x_{feature_index} | y=class_label)
        """

        feature_dist = self.feature_dists[feature_index]
        
        # categorical features
        if isinstance(x, str):
            # return probability of value for each class
            if x in feature_dist[class_label]:
                return feature_dist[class_label][x]
            elif self.smoothing_flag:  # apply laplace smoothing
                return 1 / (len(feature_dist[class_label]) + len(feature_dist))
            else:
                return 1e-9
        
        # continuous features 
        else:
            # return probability using gaussian distribution formula
            mean, variance = feature_dist[class_label]
            prob = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))
            return prob if prob != 0 else 1e-9

    def predict(self, X):
        """
        Predict labels for test matrix X

        Parameters/returns
        ----------
        :param X:  Numpy array with shape (num_samples, num_features)
        :return: Numpy array with shape (num_samples, )
            Predicted labels for each entry/row in X.
        """

        epsilon = 1e-9
        predictions = []
        
        for sample in X:
            # calculate log likelihood for each class & feature
            class_probs = {}
            for class_item in self.classes:
                class_prob = np.log(self.priors[class_item] + epsilon)
                
                for feature_index, x in enumerate(sample):
                    class_prob += \
                        np.log(self.feature_class_prob(feature_index, x, class_item) + epsilon)
                
                # store overall log likelihood for each class
                class_probs[class_item] = class_prob
            
            # predict class with highest log likelihood
            predictions.append(max(class_probs, key=class_probs.get))
        
        return np.array(predictions)


def nb_demo():
    # data from table Figure 4.8 in the textbook

    X = np.array([['Yes', 'Single', 125],
                  ['No', 'Married', 100],
                  ['No', 'Single', 70],
                  ['Yes', 'Married', 120],
                  ['No', 'Divorced', 95],
                  ['No', 'Married', 60],
                  ['Yes', 'Divorced', 220],
                  ['No', 'Single', 85],
                  ['No', 'Married', 75],
                  ['No', 'Single', 90]
                  ])

    # first two features are categorical and 3rd is continuous
    X_categorical = np.array([True, True, False])

    # class labels (default borrower)
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

    nb = NBClassifier(smoothing_flag=False)

    nb.fit(X, X_categorical, y)

    test_pts = np.array([['No', 'Married', 120],
                         ['No', 'Divorced', 95]])
    yhat = nb.predict(test_pts)

    # the book computes this as 0.0016 * alpha
    print('Predicted class for someone who is not a homeowner,')
    print('is married, and earns 120K a year is:', yhat[0])

    print('Predicted class for someone who is not a homeowner,')
    print('is divorced, and earns 95K a year is:', yhat[1])


def main():
    nb_demo()


if __name__ == "__main__":
    main()

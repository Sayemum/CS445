"""
Examples of running tf_classifers.
"""
import numpy as np
import sf_classifiers
import matplotlib.pyplot as plt


# ----------------------------------------------
# Functions for generating synthetic test data
# ----------------------------------------------

def two_clusters(num_points, noise=.3, show=False):
    """ Synthetic two-class dataset. """
    features = np.random.randint(2, size=(num_points, 1))
    features = np.append(features, features, axis=1)
    labels = np.array(np.logical_or(features[:, 0], features[:, 1]),
                      dtype=np.float32)
    features = np.array(features + np.random.normal(0, noise, features.shape),
                        dtype=np.float32)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(features[labels == 1, 0], features[labels == 1, 1], 's')
        plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'o')
        plt.show()

    return features, labels


def noisy_xor(num_points, show=False):
    """ Synthetic Dataset that is not linearly separable. """

    features = np.random.randint(2, size=(num_points, 2))
    labels = np.array(np.logical_xor(features[:, 0], features[:, 1]),
                      dtype=np.float32)
    features = np.array(features + (np.random.random(features.shape) - .5),
                        dtype=np.float32)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(features[labels == 1, 0], features[labels == 1, 1], 's')
        plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'o')
        plt.show()

    return features, labels


# ----------------------------------------------
# Examples of training models
# ----------------------------------------------


def logistic_regression_clusters(lr=.05, epochs=100):
    dataset_train_x, dataset_train_y = two_clusters(500)
    dataset_test_x, dataset_test_y = two_clusters(500)

    classifier = sf_classifiers.LogisticRegression(2)
    classifier.graph.gen_dot("logres.dot")

    classifier.train(dataset_train_x,
                     dataset_train_y,
                     epochs=epochs, learning_rate=lr)

    classifier.score(dataset_test_x, dataset_test_y)
    classifier.plot_2d_predictions(dataset_train_x, dataset_train_y)


def mlp_xor(lr=.09, epochs=100, activation='sigmoid', hidden_layers=[10]):
    dataset_train_x, dataset_train_y = noisy_xor(500)
    dataset_test_x, dataset_test_y = noisy_xor(500)

    # Create an MLP with the specified hidden layers
    classifier = sf_classifiers.MLP(2, hidden_layers, activation=activation)

    losses = classifier.train(dataset_train_x, dataset_train_y, epochs=epochs,
                     learning_rate=lr)
    
    if losses is None or not losses:  # Check if losses is empty or None
        print("Losses are empty or None")
    else:
        print("Losses: ", losses)

    classifier.score(dataset_test_x, dataset_test_y)
    classifier.plot_2d_predictions(dataset_train_x, dataset_train_y)
    
    return losses


def run_multiple_sessions(runs=10, lr=.09, epochs=100, activation='sigmoid', hidden_layers=[10]):
    all_losses = []
    
    for run in range(runs):
        losses = mlp_xor(lr=lr, epochs=epochs, activation=activation, hidden_layers=hidden_layers)
        if losses is None or not losses:
            print(f"Skipping empty losses for Run {run + 1}")
            continue
        all_losses.append(losses)
    
    return all_losses



def plot_learning_curves(all_losses, title):
    for i, losses in enumerate(all_losses):
        if losses:  # Ensure losses is not empty or None
            plt.plot(losses, label=f'Run {i + 1}')
        else:
            print(f"Skipping empty losses for Run {i + 1}")
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.legend()


def save_learning_curves(all_losses, title, filename):
    plt.figure()
    plot_learning_curves(all_losses, title)
    plt.savefig(filename)
    plt.close()  # Close the figure to avoid overlap


def ten_runs_test(deep_network=False):
    """
    Run experiments and plot learning curves.
    deep_network (bool): If True, use a deeper network (5 hidden layers).
    """
    
    # Select hidden layer configuration based on deep_network flag
    if deep_network:
        hidden_layers = [10, 10, 10, 10, 10]  # 5 hidden layers, each with 10 units
        title_suffix = "with 5 Hidden Layers"
    else:
        hidden_layers = [10]  # 1 hidden layer with 10 units (Step 4)
        title_suffix = "with 1 Hidden Layer"
    
    sigmoid_losses = run_multiple_sessions(runs=10, lr=0.09, epochs=100, activation='sigmoid', hidden_layers=hidden_layers)
    save_learning_curves(sigmoid_losses, f'Learning Curves for Sigmoid Activation {title_suffix}', f'sigmoid_{title_suffix.lower().replace(" ", "_")}.png')

    relu_losses = run_multiple_sessions(runs=10, lr=0.09, epochs=100, activation='relu', hidden_layers=hidden_layers)
    save_learning_curves(relu_losses, f'Learning Curves for ReLU Activation {title_suffix}', f'relu_{title_suffix.lower().replace(" ", "_")}.png')


if __name__ == "__main__":
    #mlp_xor()
    #logistic_regression_clusters()
    
    # To test Step 4 (1 hidden layer):
    # ten_runs_test(deep_network=False)

    # To test Step 5 (5 hidden layers):
    ten_runs_test(deep_network=True)
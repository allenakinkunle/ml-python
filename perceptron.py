import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Number of passes over the training data

    Attributes
    -----------
    weights : 1d-array
        Weights after fitting
    errors_per_epoch : list
        Number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, epochs=10):
        self.eta = eta
        self.epochs = epochs


    def fit(self, X, y):
        """Fit training data.

        Parameters
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the
            number of samples and n_features is the
            number of features.
        y : {array-like}, shape = [n_samples]
            Target values.

        Returns
        --------
        self : object
        """
        self.weights = np.zeros(1 + X.shape[1])
        self.errors_per_epoch = []

        for _ in xrange(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors_per_epoch.append(errors)

            # Stop going through the training set if the algorithm
            # converges before the number of epochs specified
            if errors == 0:
                break
        return self


    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights[1:]) + self.weights[0]


    def predict(self, X):
        """Return class label using the Heaviside step function"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

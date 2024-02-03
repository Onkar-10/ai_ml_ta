import numpy as np
import pandas as pd
from sklearn import datasets


class MultiClassLogisticRegression:

    def __init__(self, n_iter=10000, thres=1e-3):
        self.n_iter = n_iter
        self.thres = thres

    def fit(self, X, y, batch_size=64, lr=0.001, rand_seed=4, verbose=False):
        np.random.seed(rand_seed)
        self.classes = np.unique(y)
        self.class_labels = {c: i for i, c in enumerate(self.classes)}
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))
        self.fit_data(X, y, batch_size, lr, verbose)
        return self

    def fit_data(self, X, y, batch_size, lr, verbose):
        i = 0
        while (not self.n_iter or i < self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_(X)))
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.predict_(X_batch)
            update = (lr * np.dot(error.T, X_batch))
            self.weights += update
            if np.abs(update).max() < self.thres: break
            i += 1

    def predict(self, X):
        return self.predict_(self.add_bias(X))

    def predict_(self, X):
        """

        Args:
            X:

        Returns:

        """
        # Student code start TASK 1 : Write logistic function (sigmoid) for x (input scalar/array) and return it as y

        ### YOUR CODE BEGINS HERE ###
        pre_vals = np.dot(X, self.weights.T).reshape(-1, len(self.classes))
        ### YOUR CODE ENDS HERE ###

        return self.softmax(pre_vals)

    def softmax(self, z):
        """
        Apply the softmax activation function to transform the scores into probabilities.
        Args:
            z: scores - type ndarry (m number of observations, k number of classes)

        Returns: probabilities - type ndarry (m number of observations, k number of classes)

        """
        # Student code start TASK 2 : Write logistic function (sigmoid) for x (input scalar/array) and return it as y

        ### YOUR CODE BEGINS HERE ###
        post_softmax = np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)
        ### YOUR CODE ENDS HERE ###
        return post_softmax

    def predict_classes(self, X):
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))

    def add_bias(self, X):
        return np.insert(X, 0, 1, axis=1)

    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    def score(self, X, y):
        return np.mean(self.predict_classes(X) == y)

    def cross_entropy(self, y, probs):
        """
        Compute the cross entropy loss.
        Args:
            y: type ndarry (m number of observations, k number of classes)
            probs: type ndarry (m number of observations, k number of classes)

        Returns: cross entropy loss - type float

        """
        # Student code start TASK 3 : Write logistic function (sigmoid) for x (input scalar/array) and return it as y

        ### YOUR CODE BEGINS HERE ###
        post_cross_entropy = -1 * np.mean(y * np.log(probs))
        ### YOUR CODE ENDS HERE ###
        return post_cross_entropy


if __name__ == "__main__":
    data = pd.read_csv('iris.csv')
    X, y = datasets.load_iris(return_X_y=True)
    lr = MultiClassLogisticRegression(thres=1e-5)
    lr.fit(X, y, lr=0.0001)
    print(lr.score(X, y))

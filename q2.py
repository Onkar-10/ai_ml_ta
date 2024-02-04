import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import redirect_stdout


class MultiClassLogisticRegression:
    
    def __init__(self, n_iter = 10000, thres=1e-3):
        self.n_iter = n_iter
        self.thres = thres
    
    def fit(self, X, y, rand_seed=4): 
        np.random.seed(rand_seed) 
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
        return X,y
 
    def fit_data(self, X, y, batch_size=64, lr=0.001, verbose=False):
        i = 0
        while (not self.n_iter or i < self.n_iter):
            self.loss.append(self.cross_entropy(y, self.softmax(self.predict(X))))
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.softmax(self.predict(X_batch))
            update = (lr * np.dot(error.T, X_batch))
            self.weights += update
            if np.abs(update).max() < self.thres: break
            i +=1

    
    def predict(self, X):
        
             
    # Student code start TASK 1 : For each class k compute a linear combination of the input features and the weight vector of class k, that is,for each training example compute a score for each class.

    ### YOUR CODE BEGINS HERE ###
        pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
    ### YOUR CODE ENDS HERE ###
        
        return pre_vals
    
    def softmax(self, z):
    # Student code start TASK 2 : Write softmax function

    ### YOUR CODE BEGINS HERE ###
        post_softmax = np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)
    ### YOUR CODE ENDS HERE ###
        return post_softmax

  
    def add_bias(self,X):
        return np.insert(X, 0, 1, axis=1)

    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    
    def cross_entropy(self, y, probs):

    # Student code start TASK 3 : Write cross entropy loss function

    ### YOUR CODE BEGINS HERE ###
        post_cross_entropy = -1 * np.mean(y * np.log(probs))
    ### YOUR CODE ENDS HERE ###
        return post_cross_entropy
    
if __name__ == "__main__":

    data = pd.read_csv('iris.csv')
    from sklearn import datasets

    X,y = datasets.load_iris(return_X_y=True)
    lr = MultiClassLogisticRegression(thres=1e-5)
    X,y=lr.fit(X,y)
    lr.fit_data(X,y,lr=0.0001)
    print(lr.weights)
    np.savetxt('expected_output_q2.txt', lr.weights, fmt="%f")
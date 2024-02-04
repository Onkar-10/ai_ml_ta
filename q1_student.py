# Importing libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Logistic function 
def logistic(x):
    """
    Args:
        x (scalar/ndarray): scalar or numpy array of any size
    Returns:
        y (scalar/ndarray): logistic function applied to x, has the same shape as x
    """
     # Student code start TASK 1 : Write logistic function (sigmoid) for x (input scalar/array) and return it as y

    y=np.zeros_like(x)
    ### YOUR CODE BEGINS HERE ###

    ### YOUR CODE ENDS HERE ###
    return y


# Log loss
def log_loss(y, y_dash):
    """
    Args:
      y      (scalar): true value (0 or 1)
      y_dash (scalar): predicted value (probability of y being 1)
    Returns:
      loss (float): nonnegative loss corresponding to y and y_dash
    """

    loss= 0.0
    # Student code start TASK 2 : Write log loss function for inputs true value (0 or 1) and predicted value (between 0 and 1)

    ### YOUR CODE BEGINS HERE ###

    ### YOUR CODE ENDS HERE ###

    return loss


def cost_logreg(X, y, w, b):
    """
    Args:
      X (ndarray, shape (m,n))  : data on features, m observations with n features
      y (array_like, shape (m,)): array of true values of target (0 or 1)
      w (array_like, shape (n,)): weight parameters of the model      
      b (float)                 : bias parameter of the model
    Returns:
      cost (float): nonnegative cost corresponding to y and y_dash 
    """
    m, n = X.shape
    assert len(y) == m, "Number of feature observations and number of target observations do not match"
    assert len(w) == n, "Number of features and number of weight parameters do not match"

    cost=0.0
    # Student code start TASK 3 : Write Function to compute cost function in terms of data and model parameters

    ### YOUR CODE BEGINS HERE ###

    ### YOUR CODE ENDS HERE ###

    return cost


# Function to compute gradients of the cost function with respect to model parameters
def grad_logreg(X, y, w, b):
    """
    Args:
      X (ndarray, shape (m,n))  : data on features, m observations with n features
      y (array_like, shape (m,)): array of true values of target (0 or 1)
      w (array_like, shape (n,)): weight parameters of the model      
      b (float)                 : bias parameter of the model
    Returns:
      grad_w (array_like, shape (n,)): gradients of the cost function with respect to the weight parameters
      grad_b (float)                 : gradient of the cost function with respect to the bias parameter
    """
    m, n = X.shape
    assert len(y) == m, "Number of feature observations and number of target observations do not match"
    assert len(w) == n, "Number of features and number of weight parameters do not match"

    grad_w = np.zeros(n)
    grad_b=0.0
    # Student code start TASK 4a : Write Function to compute gradients of the cost function with respect to model parameters

    ### YOUR CODE BEGINS HERE ###

    ### YOUR CODE ENDS HERE ###

    
    return grad_w, grad_b


# Gradient descent algorithm for logistic regression
def grad_desc(X, y, w, b, alpha, n_iter, show_cost=True, show_params=False):
    """
    
    Args:
      X (ndarray, shape (m,n))  : data on features, m observations with n features
      y (array_like, shape (m,)): true values of target (0 or 1)
      w (array_like, shape (n,)): initial value of weight parameters
      b (scalar)                : initial value of bias parameter
      cost_func                 : function to compute cost
      grad_func                 : function to compute gradients of cost with respect to model parameters
      alpha (float)             : learning rate
      n_iter (int)              : number of iterations
    Returns:
      w (array_like, shape (n,)): updated values of weight parameters
      b (scalar)                : updated value of bias parameter
      cost_history              : List containing 
      params_history            :
    """
    m, n = X.shape
    assert len(y) == m, "Number of feature observations and number of target observations do not match"
    assert len(w) == n, "Number of features and number of weight parameters do not match"
    cost_history, params_history = [], []


    # Student code start TASK 4b : We implement batch gradient descent algorithm to learn and update model parameters with prespecified number of interations and learning rate. 
    #Just complete the 2 TO-DO tasks

    for i in range(n_iter):

        ### YOUR CODE BEGINS HERE ###
        grad_w, grad_b = grad_logreg(X, y, w, b)
        w += 0   #TODO Update 
        b += 0 #TODO Update
        cost = cost_logreg(X, y, w, b)
        ### YOUR CODE ENDS HERE ###

        cost_history.append(cost) 
        params_history.append([w, b])
        
        if show_cost == True and show_params == False and (i % math.ceil(n_iter / 10) == 0 or i == n_iter - 1):
            print(f"Iteration {i:6}:    Cost  {float(cost_history[i]):3.4f}")
        if show_cost == True and show_params == True and (i % math.ceil(n_iter / 10) == 0 or i == n_iter - 1):
            print(f"Iteration {i:6}:    Cost  {float(cost_history[i]):3.4f},    Params  {params_history[i]}")
    
            
    return w, b, cost_history, params_history


def preprocessing(filename="training.csv"):
    data = pd.read_csv(filename)
    print(pd.Series({"Dataset shape": "{}".format(data.shape)}).to_string())
    data.head()

    # Dropping unnecessary columns
    data.drop(['EventId', 'Weight'], axis=1, inplace=True)
    # Replacing -999 with nan
    data.replace(to_replace=-999, value=np.nan, inplace=True)
    # Encoding the 'Label' column
    label_dict = {'b': 0, 's': 1}
    data.replace({'Label': label_dict}, inplace=True)
    # Train-test split
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=40)
    # Columns with missing values with respective proportions
    (data.isna().sum()[data.isna().sum() > 0] / len(data)).sort_values(ascending=False)
    # Discarding columns with more than 30% missing data
    cols_missing_drop = [
        'DER_deltaeta_jet_jet',
        'DER_mass_jet_jet',
        'DER_prodeta_jet_jet',
        'DER_lep_eta_centrality',
        'PRI_jet_subleading_pt',
        'PRI_jet_subleading_eta',
        'PRI_jet_subleading_phi',
        'PRI_jet_leading_pt',
        'PRI_jet_leading_eta',
        'PRI_jet_leading_phi'
    ]
    data_train.drop(cols_missing_drop, axis=1, inplace=True)
    data_test.drop(cols_missing_drop, axis=1, inplace=True)
    # Median imputation
    data_train['DER_mass_MMC'].fillna(data_train['DER_mass_MMC'].median(), inplace=True)
    data_test['DER_mass_MMC'].fillna(data_test['DER_mass_MMC'].median(), inplace=True)
    # Features-target split
    X_train, y_train = data_train.drop('Label', axis=1), data_train['Label']
    X_test, y_test = data_test.drop('Label', axis=1), data_test['Label']
    # Min-Max normalization
    for col in X_train.columns:
        if (X_train[col].dtypes == 'int64' or X_train[col].dtypes == 'float64') and X_train[col].nunique() > 1:
            X_train[col] = (X_train[col] - X_train[col].min()) / (X_train[col].max() - X_train[col].min())
    for col in X_test.columns:
        if (X_test[col].dtypes == 'int64' or X_test[col].dtypes == 'float64') and X_test[col].nunique() > 1:
            X_test[col] = (X_test[col] - X_test[col].min()) / (X_test[col].max() - X_test[col].min())
    
    return X_train,X_test,y_train,y_test



# Function to compute confusion matrix
def conf_mat(y_test, y_pred):
    """
    Computes confusion matrix
    Args:
      y_test (array_like): true binary (0 or 1) labels
      y_pred (array_like): predicted binary (0 or 1) labels
    Returns:
      confusion_mat (array): A 2D array representing a 2x2 confusion matrix
    """
    y_test, y_pred = list(y_test), list(y_pred)
    count, labels, confusion_mat = len(y_test), [0, 1], np.zeros(shape=(2, 2), dtype=int)
    for i in range(2):
        for j in range(2):
            confusion_mat[i][j] = len([k for k in range(count) if y_test[k] == labels[i] and y_pred[k] == labels[j]])
    return confusion_mat


# Function to compute accuracy
def accuracy(y_test, y_pred):
    """
    Computes accuracy, given true and predicted binary (0 or 1) labels
    Args:
      y_test (array_like): true binary (0 or 1) labels
      y_pred (array_like): predicted binary (0 or 1) labels
    Returns:
      acc (float): accuracy obtained from y_test and y_pred
    """
    confusion_mat = conf_mat(y_test, y_pred)
    num = confusion_mat[0, 0] + confusion_mat[1, 1]  # Number of correct predictions
    denom = num + confusion_mat[0, 1] + confusion_mat[1, 0]  # Number of total predictions
    acc = num / denom
    return acc


if __name__ == "__main__":
    # Loading the data
    
    # Initial values of the model parameters
    X_train,X_test,y_train,y_test=preprocessing("training.csv")
    w_init = np.array([-5, -15, -10, 9, 4, -6, 3, -10, 1, 14, 0, 0, 15, 0, 0, 7, 0, -3, 1, -8]).astype(float)
    b_init = -1.

    # Learning model parameters using gradient descent algorithm
    # Change n_iter as per convenience
    w_out, b_out, cost_history, params_history = grad_desc(X_train.to_numpy(),
                                                           y_train.to_numpy(),
                                                           w=w_init,  # np.zeros(X_train.shape[1]),
                                                           b=b_init,  # 0,
                                                           alpha=0.1,
                                                           n_iter=40)

    # Prediction and evaluation on the training set and the test set
    y_train_prob = logistic(np.matmul(X_train.to_numpy(), w_out) + (b_out * np.ones(X_train.shape[0])))
    y_test_prob = logistic(np.matmul(X_test.to_numpy(), w_out) + (b_out * np.ones(X_test.shape[0])))
    y_train_pred, y_test_pred = (y_train_prob > 0.5).astype(int), (y_test_prob > 0.5).astype(int)
    print(pd.Series({"Training accuracy": accuracy(y_train, y_train_pred),
                     "Test accuracy": accuracy(y_test, y_test_pred)}).to_string())
    pass
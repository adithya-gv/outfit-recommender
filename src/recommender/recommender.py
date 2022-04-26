from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train():
    X = pd.read_csv("data/training/articles_subset.csv")[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']] # you can add more features
    y = pd.read_csv("data/training/articles_subset.csv")[['cluster']] # modify this to your target variable
    X = X.to_numpy()
    y = y.to_numpy()
    N,_ = y.shape
    y = y.reshape((N,))

    for i in range(len(X[:, 6])):
        X[i, 6] = ord(X[i, 6]) - ord('A')

    X_train = X[:int(0.8 * len(X))]
    y_train = y[:int(0.8 * len(y))]
    X_test = X[int(0.8 * len(X)):]
    y_test = y[int(0.8 * len(y)):]

    model = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    model = model.fit(X_train, y_train)
    return model

def recommend(X):
    for i in range(len(X[:, 6])):
        X[i, 6] = ord(X[i, 6]) - ord('A')
    
    model = train()
    
    return model.kneighbors(X)

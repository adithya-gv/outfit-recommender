import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
 
X = pd.read_csv("data/training/articles_subset.csv")[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']] # you can add more features
y = pd.read_csv("data/training/articles_subset.csv")[['cluster']] # modify this to your target variable
X = X.to_numpy()
y = y.to_numpy()

for i in range(len(X[:, 6])):
    X[i, 6] = ord(X[i, 6]) - ord('A')
 
X = X.astype(np.float32)
y = y.astype(np.float32)

X_train = X[:int(0.8 * len(X))]
y_train = y[:int(0.8 * len(y))]
X_test = X[int(0.8 * len(X)):]
y_test = y[int(0.8 * len(y)):]

X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)
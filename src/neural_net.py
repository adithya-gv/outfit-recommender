import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
 
X = pd.read_csv("data/training/dataset.csv")[['graphical_appearance_no', 'colour_group_code']]
y = pd.read_csv("data/training/dataset.csv")[['cluster']]
X = X.to_numpy()
y = y.to_numpy()

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
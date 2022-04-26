import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

X = pd.read_csv("data/training/articles_subset.csv")[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']] # you can add more features
y = pd.read_csv("data/training/articles_subset.csv")[['cluster']] # modify this to your target variable
X = X.to_numpy()
y = y.to_numpy()
N,_ = y.shape
y =y.reshape((N,1))
for i in range(len(X[:, 6])):
    X[i, 6] = ord(X[i, 6]) - ord('A')
 
for i in range(np.shape(X)[1]) :
    X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    
real_y = np.zeros(shape=(N, 15))
    
for i in range(N):
    real_y[i, y[i, 0]] = 1

X = X.astype(np.float32)
real_y = real_y.astype(np.float32)


X_train = X[:int(0.8 * len(X))]
y_train = y[:int(0.8 * len(y))]
X_test = X[int(0.8 * len(X)):]
y_test = y[int(0.8 * len(y)):]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='elu'),
    tf.keras.layers.Dense(55, activation='elu'),
    tf.keras.layers.Dense(35, activation='elu'),
    tf.keras.layers.Dense(15, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=40)
model.evaluate(X_test, y_test, verbose=2)

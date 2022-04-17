import pandas as pd
import numpy as np

def preprocess_articles():
    dfs = []
    for i in range(11, 65):
        if not (i == 13 or i == 16 or i == 57):
            df = pd.read_csv('results/modified/0' + str(i) + '.csv')
            dfs.append(df)
    result = pd.concat(dfs)
    articles = pd.read_csv("data/articles.csv")
    images = articles.to_numpy()[:, 0]
    indices = result.to_numpy()[:, 0]
    splice = []
    for i in indices:
        index = np.where(images == i)[0][0]
        splice.append(index)
        print(index)
    real_articles = articles.iloc[splice]
    clusters = result.to_numpy()[:, 1]
    real_articles.insert(25, 'cluster', clusters)
    path = "data/articles_subset.csv"
    real_articles.to_csv(path, index=False)

def preprocess_customers():
    dfs = []
    for i in range(11, 65):
        if not (i == 13 or i == 16 or i == 57):
            df = pd.read_csv('results/modified/0' + str(i) + '.csv')
            dfs.append(df)
    result = pd.concat(dfs)
    customers = pd.read_csv("data/transactions_train.csv")
    c = customers.to_numpy()[180001:200000, 2]
    print(c[0])
    indices = result.to_numpy()[:, 0]
    splice = []
    for i in indices:
        index = np.where(c == i)
        if (np.shape(index)[0] > 0):
            indices = index[0]
            for s in indices:
                splice.append(180001 + s)
                print(180001 + s)
    real_customers = customers.iloc[splice]
    path = "data/training/training_data_10.csv"
    real_customers.to_csv(path, index=False)


preprocess_customers()
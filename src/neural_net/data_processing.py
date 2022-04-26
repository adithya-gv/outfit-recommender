import pandas as pd
import numpy as np

# removes articles that were not clustered
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

# pulled purchase history of first 1000 customers.
def preprocess_customers():
    result = pd.read_csv("data/customers.csv")
    customers = pd.read_csv("data/transactions_train.csv")
    c = customers.to_numpy()[:, 1]
    print(c[0])
    indices = result.to_numpy()[:, 0]
    for i in range(0, 1000):
        splice = []
        print(i)
        index = np.where(c == indices[i])
        if (np.shape(index)[0] > 0):
            k = index[0]
            for s in k:
                splice.append(s)
        real_customers = customers.iloc[splice]
        path = "data/customer_samples/customer_data_" + str(i) + ".csv"
        real_customers.to_csv(path, index=False)

# concatenate all of the 1000 customers into one dataset
def create_data():
    result_path = "data/training/data.csv"
    articles = pd.read_csv("data/training/articles_subset.csv")
    articles = articles.to_numpy().flatten()
    datas = []
    for k in range(0, 1000):
        print(k)
        path = "data/customer_samples/customer_data_" + str(k) + ".csv"
        df = pd.read_csv(path)
        A = df.to_numpy()[:, 2]
        indices = []
        for article in articles:
            index = np.where(A == article)[0]
            if (np.shape(index)[0] > 0):
                for i in index:
                    indices.append(i)
        if (np.shape(indices)[0] > 0):
            datas.append(df.iloc[indices])
    actual_data = pd.concat(datas)
    actual_data.to_csv(result_path, index=False)

# this is a test function used to generate arbitrary purchase histories for a customer. we can play around with this. 
def create_vectors():
    articles = pd.read_csv("data/training/data.csv")
    articles = articles.to_numpy()[:, 2]
    articles = np.unique(articles)
    vectors = np.ndarray(shape=(len(articles), 2))
    i = 0
    for article in articles:
        print(i)
        vector = np.zeros(1)
        if (np.random.random() > 0.5):
            vector[0] = 1
        vector = np.concatenate(([article], vector))
        vectors[i] = vector
        i += 1
    vectors = pd.DataFrame(vectors, dtype=int)
    vectors.to_csv("data/training/vectors.csv", index=False)
            
# appends purchase history to the customer dataset and generates training dataset
def create_dataset():
    vectors = pd.read_csv("data/training/vectors.csv")
    new_vectors = vectors.drop(columns=['0'])
    articles = vectors.to_numpy()[:, 0]
    real_articles = pd.read_csv("data/training/articles_subset.csv")
    ra = real_articles.to_numpy()[:, 0]
    splice = [] 
    for article in articles:
        print(article)
        index = np.where(ra == article)[0]
        if (np.shape(index)[0] > 0):
            splice.append(index[0])
    final_articles = real_articles.iloc[splice]
    final_articles.reset_index(drop=True, inplace=True)
    pd.concat([final_articles, new_vectors], ignore_index=True, axis=1).to_csv("data/training/dataset.csv", index=False)


create_vectors()
create_dataset()
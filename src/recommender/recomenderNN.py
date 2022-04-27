from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

articles = pd.read_csv("data/articles.csv")[['article_id']].to_numpy() # the entire images data file

def train(i):
    X = pd.read_csv("data/cluster_partitions/cluster" + str(i) + ".csv")[['graphical_appearance_no', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']] # you can add more features
    y = pd.read_csv("data/cluster_partitions/cluster" + str(i) + ".csv")[['product_type_no']] # modify this to your target variable
    X = X.to_numpy()
    y = y.to_numpy()
    N,_ = y.shape
    y = y.reshape((N,))

    for i in range(len(X[:, 3])):
        X[i, 3] = ord(X[i, 3]) - ord('A')

    X_train = X[:int(0.8 * len(X))]
    y_train = y[:int(0.8 * len(y))]
    X_test = X[int(0.8 * len(X)):]
    y_test = y[int(0.8 * len(y)):]

    model = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
    model = model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    return model

def recommend(X):
    path = "results/articles_clustered.csv"
    art = pd.read_csv(path).to_numpy()

    for i in range(len(X[:, 4])):
        X[i, 4] = ord(X[i, 4]) - ord('A')
    
    for i in X:
        a = art[np.where(art[:, 1] == i[0])]
        cluster = a[:, 12][0]
        model = train(cluster)
    
    return model.kneighbors(X[:, 1:])

def show_image(indexes):
    for i, index in enumerate(indexes):
        id = "0" + str(articles[index][0])
        input_img = mpimg.imread("data/images/" + id[0:3] + "/" + id + ".jpg")
        a = fig.add_subplot(2, 3, i+2)
        imgplot = plt.imshow(input_img)


for i in range(0, 5):
    path = "data/customer_samples/customer_data_" + str(i) + ".csv"
    allData = pd.read_csv(path).to_numpy()
    customerPurchases = pd.read_csv(path)[['article_id', 'graphical_appearance_no', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']]
    #customerPurchases = pd.read_csv(path)[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'department_no', 'index_group_no', 'section_no', 'garment_group_no']]


    customerPurchases = customerPurchases.to_numpy()
    allRecommendations = recommend(customerPurchases)[1]
    customerPurchases = pd.read_csv(path)[['product_type_no', 'graphical_appearance_no', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']]
    customerPurchases = customerPurchases.to_numpy()
    for i, rec in enumerate(allRecommendations):
        print(rec)
        fig = plt.figure()
        show_image(rec)
        a = fig.add_subplot(2, 3, 1)
        id  = "0" + str(allData[i, 0])
        print(id)
        try:
            imgplot = plt.imshow(mpimg.imread("data/images/" + id[0:3] + "/" + id + ".jpg"))
        except FileNotFoundError:
            continue
        plt.show()

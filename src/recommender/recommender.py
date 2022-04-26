from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

articles = pd.read_csv("data/articles.csv")[['article_id']].to_numpy() # the entire images data file

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

"""
Takes in a list of indexes and displays all the images that they correspond to
"""
def show_image(indexes):
    fig = plt.figure()
    for i, index in enumerate(indexes):
        id = "0" + str(articles[index][0])
        input_img = mpimg.imread("data/images/" + id[0:3] + "/" + id + ".jpg")
        a = fig.add_subplot(2, 3, i+1)
        imgplot = plt.imshow(input_img)
    plt.show()


for i in range(5):
    path = "data/customer_samples/customer_data_" + str(i) + ".csv"
    customerPurchases = pd.read_csv(path)[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']]
    customerPurchases = customerPurchases.to_numpy()
    allRecommendations = recommend(customerPurchases)[1]
    for i in allRecommendations:
        show_image(i)


# input_index = [140] #this is the input image that you want to find similar items to 
# input = pd.read_csv("data/articles.csv")[['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']].to_numpy()[input_index[0]]
# input = input.reshape((1, input.shape[0]))
# show_image(input_index) # displays the input image

# indexes = recommend(input)
# output_indexes = indexes[1][0]
# show_image(output_indexes) # displays all 5 output images

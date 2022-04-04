import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm

from clustering import Clustering

subdir = "094" # change this

labels = []
var = []

# KMeans PCA Analysis  
def KMeans_PCA_Variance_Ratio(grayscale=False):
    labels = []
    var = []
    mean = 0
    k = 2
    while mean < 0.91:
        print("Running KMeans with PCA image compression using", k, "components.")
        k_clustering = Clustering(K=k, analysis=1)
        _, _, mean = k_clustering.generate_images_overwrite(u_subdir=subdir, grayscale=grayscale)
        print(mean)
        labels.append(k)
        k = k + 1
        var.append(mean)
    
    plt.plot(labels, var)
    plt.title('Recovered Variance Ratio vs. PCA components')
    plt.xlabel('Components Retained')
    plt.ylabel('Recovered Variance Ratio')
    plt.show()

def KMeans_Optimal_Clusters(grayscale=False):
    labels = []
    objective_values = []
    for k in range(2, 25):
        print(k)
        k_clustering = Clustering(K=16, num_clusters=k)
        _, inertia_k, _ = k_clustering.cluster_images_kmeans(u_subdir=subdir, grayscale = grayscale)
        labels.append(k)
        objective_values.append(inertia_k)
    
    plt.plot(labels, objective_values)
    plt.title('Cluster Count vs. Inertia')
    plt.xlabel('Clusters Used')
    plt.ylabel('Inertia')
    plt.show()

def Compute_DaviesBouldinIndex():
    clustering = Clustering(K=16, num_clusters=13)
    k_db = []
    a_db = []
    k_ss = []
    a_ss = []
    for i in range(0, 10):
        labels_k, _, images = clustering.cluster_images_kmeans_dataset(u_subdir=subdir, grayscale=False)
        labels_a, _ = clustering.cluster_images_agglomerate(u_subdir=subdir)

        db_k = skm.davies_bouldin_score(images, labels_k)
        db_a = skm.davies_bouldin_score(images, labels_a)
        print(db_k)
        print(db_a)
        k_db.append(db_k)
        a_db.append(db_a)

        ss_k = skm.silhouette_score(images, labels_k)
        ss_a = skm.silhouette_score(images, labels_a)
        print(ss_k)
        print(ss_a)
        k_ss.append(ss_k)
        a_ss.append(ss_a)
    
    print()
    print()
    print("K-Means Davies Bouldin Index:", np.mean(k_db))
    print("Agglomerate Davies Bouldin Index:", np.mean(a_db))
    print()
    print("K-Means Silhouette Score:", np.mean(k_ss))
    print("Agglomerate Silhouette Score:", np.mean(a_ss))

## Function to Run
Compute_DaviesBouldinIndex()

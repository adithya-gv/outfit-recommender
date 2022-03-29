import os

import matplotlib.pyplot as plt
import numpy as np

from clustering import Clustering

subdir = "028" # change this

labels = []
var = []

# KMeans PCA Analysis  
def KMeans_PCA_Variance_Ratio(k):
    print("Running KMeans with PCA image compression using", k, "components.")
    k_clustering = Clustering(K=k,num_clusters=5, analysis=1)

    labels_k = k_clustering.cluster_images_kmeans(u_subdir=subdir)

def plot_variance_ratio():
    f = open("results/variance.txt", "r")

    count = 2
    labels = []
    var = []
    for x in f:
        labels.append(count)
        var.append(float(x))
        count = count + 1
    plt.plot(labels, var)
    plt.title('Recovered Variance Ratio vs. PCA components')
    plt.xlabel('Components Retained')
    plt.ylabel('Recovered Variance Ratio')
    plt.show()

def KMeans_Optimal_Clusters():
    labels = []
    objective_values = []
    for k in range(2, 16):
        k_clustering = Clustering(num_clusters=k)
        _, inertia_k = k_clustering.cluster_images_kmeans(u_subdir=subdir)
        labels.append(k)
        objective_values.append(inertia_k)
    
    plt.plot(labels, objective_values)
    plt.title('Cluster Count vs. Inertia')
    plt.xlabel('Clusters Used')
    plt.ylabel('Inertia')
    plt.show()



## Function to Run
KMeans_Optimal_Clusters()
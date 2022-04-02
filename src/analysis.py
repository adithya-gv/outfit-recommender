import os

import matplotlib.pyplot as plt
import numpy as np

from clustering import Clustering

subdir = "094" # change this

labels = []
var = []

# KMeans PCA Analysis  
def KMeans_PCA_Variance_Ratio():
    labels = []
    var = []
    mean = 0
    k = 2
    while mean < 0.91:
        print("Running KMeans with PCA image compression using", k, "components.")
        k_clustering = Clustering(K=k, analysis=1)

        _, _, mean = k_clustering.generate_images_overwrite(u_subdir=subdir)
        print(mean)
        labels.append(k)
        k = k + 1
        var.append(mean)
    
    plt.plot(labels, var)
    plt.title('Recovered Variance Ratio vs. PCA components')
    plt.xlabel('Components Retained')
    plt.ylabel('Recovered Variance Ratio')
    plt.show()

def KMeans_Optimal_Clusters():
    labels = []
    objective_values = []
    for k in range(2, 25):
        print(k)
        k_clustering = Clustering(K=16, num_clusters=k)
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
from cProfile import label
import os

import matplotlib.pyplot as plt
import numpy as np

from clustering import IMAGES_PATH, Clustering

agg_clustering = Clustering(num_clusters=10)
k_clustering = Clustering(K = 16, num_clusters=13)

subdir = "030" # change this

# Setup for KMeans
labels_k, _, paths = k_clustering.cluster_images_kmeans(u_subdir=subdir)

images = np.load(os.path.join(IMAGES_PATH, subdir, "all_images.npy"))
images = images.reshape(len(images), 250, 250, 3)

# Setup for Agglomerate Clustering
"""
labels_a = agg_clustering.cluster_images_agglomerate(u_subdir=subdir)
"""
path = "results/data/cluster_results_" + subdir + ".csv"
f = open(path, "w")
k = 0
for i in labels_k:
    string = str(paths[k].split("/")[3]) + "," + str(i) + "\n"
    k = k + 1
    f.write(string)
f.close()

# Setup for Plot
"""
for img in images:
    ax = fig.add_subplot(2, len(images), i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    ax.set_title(f'{labels_a[i]}')
    i += 1
"""

plt.show()

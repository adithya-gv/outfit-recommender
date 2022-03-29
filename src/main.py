import os

import matplotlib.pyplot as plt
import numpy as np

from clustering import IMAGES_PATH, Clustering

agg_clustering = Clustering(num_clusters=10)
k_clustering = Clustering(num_clusters=10)

subdir = "028" # change this

# Setup for KMeans
labels_k, _ = k_clustering.cluster_images_kmeans(u_subdir=subdir)

images = np.load(os.path.join(IMAGES_PATH, subdir, "all_images.npy"))
images = images.reshape(len(images), 1000, 1000, 3)

"""
# Setup for Agglomerate Clustering
labels_a = agg_clustering.cluster_images_agglomerate(u_subdir=subdir)
"""

fig = plt.figure(figsize=(2, len(images)))
i = 0

for img in images:
    ax = fig.add_subplot(2, len(images), i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    ax.set_title(f'{labels_k[i - len(images)]}')
    i += 1

"""
for img in images:
    ax = fig.add_subplot(2, len(images), i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    ax.set_title(f'{labels_a[i]}')
    i += 1
"""

plt.show()

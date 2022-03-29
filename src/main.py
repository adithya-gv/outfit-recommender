import os

import matplotlib.pyplot as plt
import numpy as np

from clustering import IMAGES_PATH, Clustering

clustering = Clustering(num_clusters=10)
subdir = "020"
labels_a = clustering.cluster_images_agglomerate(u_subdir=subdir)
# labels_k = clustering.cluster_images_kmeans(u_subdir=subdir)

images = np.load(os.path.join(IMAGES_PATH, subdir, "all_images.npy"))
images = images.reshape(len(images), 1000, 1000, 3)

fig = plt.figure(figsize=(2, len(images)))
i = 0
for img in images:
    ax = fig.add_subplot(2, len(images), i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    ax.set_title(f'{labels_a[i]}')
    i += 1

# for img in images:
#     ax = fig.add_subplot(2, len(images), i + 1, xticks=[], yticks=[])
#     ax.imshow(img)
#     ax.set_title(f'{labels_k[i - len(images)]}')
#     i += 1

plt.show()

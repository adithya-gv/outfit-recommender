from cProfile import label
import os

import matplotlib.pyplot as plt
import numpy as np

from clustering import IMAGES_PATH, Clustering

agg_clustering = Clustering(K = 16, num_clusters=15) #15 clusters for grayscale
k_clustering = Clustering(K = 16, num_clusters=13)

subdir = "000" # change this


# Note that folder 057 is not fully done to an error, will have to resolve later. Not sure if it was 
# running out of RAM or that image just couldn't be clustered for some reason (0578382006)

# Setup for KMeans (loops over multiple folders)
# for i in range(58, 90):
#     subdir = "0" + str(i)
#     labels_k, _, paths = k_clustering.cluster_images_kmeans(u_subdir=subdir)

#     path = "results/data/cluster_results_" + subdir + ".csv"
#     f = open(path, "w")
#     k = 0
#     for i in labels_k:
#         string = str(paths[k].split("\\")[2]) + "," + str(i) + "\n" # this for windows
#         #string = str(paths[k].split("/")[3]) + "," + str(i) + "\n" # this for mac/linux
#         k = k + 1
#         f.write(string)
#     f.close()



# images = np.load(os.path.join(IMAGES_PATH, subdir, "all_images.npy"))
# images = images.reshape(len(images), 250, 250, 3)


# Setup for Agglomerate Clustering

labels_a, paths = agg_clustering.cluster_images_agglomerate(u_subdir=subdir)
path = "results/data/batch_cluster_results.csv"
f = open(path, "w")
f.write("image,cluster\n")
k = 0
for i in labels_a:
    print(paths[k].split("/")[3])
    # string = str(paths[k].split("\\")[2]) + "," + str(i) + "\n" # this for windows
    string = str(paths[k].split("/")[3]) + "," + str(i) + "\n" # this for mac/linux
    k = k + 1
    f.write(string)
f.close()

"""
for i in range(14, 65):
    subdir = "0" + str(i) + "_grayscale"
    if subdir != "013_grayscale" and subdir != "016_grayscale": # these folders don't have enough images to cluster
        labels_a, paths = agg_clustering.cluster_images_agglomerate(u_subdir=subdir, grayscale = True)

        path = "results/data/cluster_results_" + "grayscale_" + subdir + ".csv"
        f = open(path, "w")
        k = 0
        for i in labels_a:
            # string = str(paths[k].split("\\")[2]) + "," + str(i) + "\n" # this for windows
            string = str(paths[k].split("/")[3]) + "," + str(i) + "\n" # this for mac/linux
            k = k + 1
            f.write(string)
        f.close()
        print("clustered images in folder " + subdir)
"""

# Setup for Plot
#fig = plt.figure(figsize=(2, len(images)))


# for i, img in enumerate(images):
#     ax = fig.add_subplot(2, len(images), i + 1, xticks=[], yticks=[])
#     ax.imshow(img)
#     ax.set_title(f'{labels_k[i]}')
#     i += 1

# plt.show()

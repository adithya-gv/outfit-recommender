from operator import iadd
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class Clustering:
    def __init__(self, K=5, num_clusters=8):
        self.K = K
        self.num_clusters = num_clusters

    def __rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

    def __compress_image(self, subdir, file):
        path_to_file = os.path.join(subdir, file)
        save_file_path = os.path.splitext(path_to_file)[0] + ".npy"

        if not os.path.exists(save_file_path):
            img = cv2.cvtColor(cv2.imread(path_to_file), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (1000, 1000))
            blue, green, red = cv2.split(img)

            df_blue = blue / 255.
            df_green = green / 255.
            df_red = red / 255.

            pca_b = PCA(n_components=self.K)
            trans_pca_b = pca_b.fit_transform(df_blue)

            pca_g = PCA(n_components=self.K)
            trans_pca_g = pca_g.fit_transform(df_green)

            pca_r = PCA(n_components=self.K)
            trans_pca_r = pca_r.fit_transform(df_red)

            b_arr = pca_b.inverse_transform(trans_pca_b)
            g_arr = pca_g.inverse_transform(trans_pca_g)
            r_arr = pca_r.inverse_transform(trans_pca_r)

            new_img = np.clip(cv2.merge((b_arr, g_arr, r_arr)), 0, 1)

            np.save(save_file_path, new_img)
            print(f'Created compressed image for {path_to_file}')
        
        return np.load(save_file_path, encoding='bytes')

    def __generate_images(self, u_subdir=""):
        for subdir, dirs, files in os.walk(os.path.join("data/images", u_subdir)):
            files = [ fi for fi in files if fi.endswith(".jpg") ]
            imgs = np.zeros((len(files), 1000, 1000, 3))
            i = 0
            for file in files:
                imgs[i, ...] = self.__compress_image(subdir, file)
                i += 1
        return imgs

    def cluster_images(self, u_subdir="", display = False):
        original_images = self.__generate_images(u_subdir=u_subdir)
        images = original_images.reshape(len(original_images), -1)
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(images)

        labels = kmeans.labels_

        if display:
            fig = plt.figure(figsize=(1, len(images)))
            i = 0
            for img in original_images:
                ax = fig.add_subplot(1, len(images), i + 1, xticks=[], yticks=[])
                ax.imshow(img)
                ax.set_title(f'{labels[i]}')
                i += 1
            
            plt.show()

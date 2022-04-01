from operator import iadd
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

IMAGES_PATH = "data/images"

class Clustering:
    def __init__(self, K=5, num_clusters=8, analysis=0):
        self.K = K
        self.num_clusters = num_clusters
        self.analysis = analysis

    def __rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

    def __compress_image(self, subdir, file):
        path_to_file = os.path.join(subdir, file)
        save_file_path = os.path.splitext(path_to_file)[0] + ".npy"

        if not os.path.exists(save_file_path):
            img = cv2.cvtColor(cv2.imread(path_to_file), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (250, 250))
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
        
        return np.load(save_file_path, encoding='bytes'), os.path.splitext(path_to_file)[0]

    def __generate_images(self, u_subdir=""):
        all_images = []
        paths = []
        total_num_files = 0
        for subdir, dirs, files in os.walk(os.path.join(IMAGES_PATH, u_subdir)):
            files = [ fi for fi in files if fi.endswith(".jpg") ]
            imgs = np.zeros((len(files), 250, 250, 3))
            i = 0
            for file in files:
                imgs[i, ...], path = self.__compress_image(subdir, file)
                i += 1
                paths.append(path)
            total_num_files += i
            all_images.append((imgs, subdir))
        return all_images, total_num_files, paths

    def __compress_image_overwrite(self, subdir, file):
        path_to_file = os.path.join(subdir, file)
        save_file_path = os.path.splitext(path_to_file)[0] + ".npy"

        img = cv2.cvtColor(cv2.imread(path_to_file), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (250, 250))
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

        if (self.analysis == 1):
            mean = (pca_r.explained_variance_ratio_ + pca_b.explained_variance_ratio_ + pca_g.explained_variance_ratio_) / 3
            mean = np.sum(mean)

        b_arr = pca_b.inverse_transform(trans_pca_b)
        g_arr = pca_g.inverse_transform(trans_pca_g)
        r_arr = pca_r.inverse_transform(trans_pca_r)

        new_img = np.clip(cv2.merge((b_arr, g_arr, r_arr)), 0, 1)

        np.save(save_file_path, new_img)        
        return np.load(save_file_path, encoding='bytes'), mean

    def generate_images_overwrite(self, u_subdir=""):
        all_images = []
        total_num_files = 0
        sum = 0.0
        for subdir, dirs, files in os.walk(os.path.join(IMAGES_PATH, u_subdir)):
            files = [ fi for fi in files if fi.endswith(".jpg") ]
            imgs = np.zeros((len(files), 250, 250, 3))
            i = 0
            for file in files:
                imgs[i, ...], mean = self.__compress_image_overwrite(subdir, file)
                sum = sum + mean
                i += 1
            total_num_files += i
            all_images.append((imgs, subdir))
        
        return all_images, total_num_files, (sum / total_num_files)

    def cluster_images_kmeans(self, u_subdir=""):
        all_images_path = os.path.join(IMAGES_PATH, u_subdir, "all_images.npy")
        if not os.path.exists(all_images_path):
            original_images, total_num_images, paths = self.__generate_images(u_subdir=u_subdir)

            all_images = np.zeros((total_num_images, 250 * 250 * 3))

            start = 0
            for image_subdir, subdir in original_images:
                end = start + len(image_subdir)
                all_images[start:end] = image_subdir.reshape(len(image_subdir), -1)
                start = end

            np.save(all_images_path, all_images)

        images = np.load(all_images_path)
        kmeans = KMeans(n_clusters=self.num_clusters)

        kmeans.fit(images)

        return kmeans.labels_, kmeans.inertia_, paths
    
    def cluster_images_agglomerate(self, u_subdir=""):
        all_images_path = os.path.join(IMAGES_PATH, u_subdir, "all_images.npy")
        if not os.path.exists(all_images_path):
            original_images, total_num_images = self.__generate_images(u_subdir=u_subdir)

            all_images = np.zeros((total_num_images, 250 * 250 * 3))

            start = 0
            for image_subdir, subdir in original_images:
                end = start + len(image_subdir)
                all_images[start:end] = image_subdir.reshape(len(image_subdir), -1)
                start = end
        
            np.save(all_images_path, all_images)

        images = np.load(all_images_path)
        c = AgglomerativeClustering(n_clusters=self.num_clusters, compute_distances=True)

        c.fit(images)

        return c.labels_, c.distances_

    def cluster_images_varied_agglomerate(self, u_subdir="", distance_threshold=0.5):
        all_images_path = os.path.join(IMAGES_PATH, u_subdir, "all_images.npy")
        if not os.path.exists(all_images_path):
            original_images, total_num_images = self.__generate_images(u_subdir=u_subdir)

            all_images = np.zeros((total_num_images, 1000 * 1000 * 3))

            start = 0
            for image_subdir, subdir in original_images:
                end = start + len(image_subdir)
                all_images[start:end] = image_subdir.reshape(len(image_subdir), -1)
                start = end
        
            np.save(all_images_path, all_images)

        images = np.load(all_images_path)
        c = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)

        c.fit(images)

        return c.labels_, c.n_clusters_

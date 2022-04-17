import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
import shutil

from clustering import Clustering

def preprocess():
    for i in range(11, 65):
        print()
        print(i)
        print()
        images = []
        if not (i == 13 or i == 16 or i == 57):
            path = "results/data/cluster_results_agg_0" + str(i) + ".csv"
            df = pd.read_csv(path)
            arr_0 = df.to_numpy()[:, 0]
            arr_1 = df.to_numpy()[:, 1]
            clusterFound = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for j in range(len(arr_1)):
                if (np.sum(clusterFound) == 13):
                    break
                if clusterFound[arr_1[j]] == 1:
                    continue
                else:
                    clusterFound[arr_1[j]] = 1
                    print(arr_0[j])
                    images.append("0" + str(arr_0[j]) + ".jpg")
        for image in images:
            src = "data/images/0" + str(i) + "/" + image
            dst = "data/images/000/" + image
            shutil.copyfile(src, dst)

def postprocess():
    path = "results/data/batch_cluster_results.csv"
    df = pd.read_csv(path)
    titles = df.columns
    arr_0 = df.to_numpy()[:, 0]
    arr_1 = df.to_numpy()[:, 1]
    map = {}
    map["044"] = [(8, 2)]
    for i in range(len(arr_0)):
        dir = str(arr_0[i])[0:2]
        path = "results/data/cluster_results_agg_0" + dir + ".csv"
        df_t = pd.read_csv(path)
        arr_0t = df_t.to_numpy()[:, 0]
        arr_1t = df_t.to_numpy()[:, 1]
        new_cluster = arr_1[i]
        for j in range(len(arr_0t)):
            if arr_0[i] == arr_0t[j]:
                old_cluster = arr_1t[j]
                if ("0" + str(arr_0[i])[0:2]) in map:
                    map[("0" + str(arr_0[i])[0:2])].append((new_cluster, old_cluster))
                else:
                    map["0" + str(arr_0[i])[0:2]] = [(new_cluster, old_cluster)]
                break
    print(map)
    gen_new_map(map)

def gen_new_map(map):
    for i in range(11, 65):
        print(i)
        images = []
        if not (i == 13 or i == 16 or i == 57):
            path = "results/data/cluster_results_agg_0" + str(i) + ".csv"
            arr = map["0" + str(i)]
            new_map = {}
            for a in arr:
                new_map[a[1]] = a[0] # map old cluster to new cluster
            df = pd.read_csv(path)
            img = df.to_numpy()[:, 0]
            old_cluster = df.to_numpy()[:, 1]
            new_path = "results/modified/0" + str(i) + ".csv"
            f = open(new_path, "w")
            for i in range(len(img)):
                oc = old_cluster[i]
                nc = new_map[oc]
                f.write(str(img[i]) + "," + str(nc) + "\n")
            f.close()


postprocess()

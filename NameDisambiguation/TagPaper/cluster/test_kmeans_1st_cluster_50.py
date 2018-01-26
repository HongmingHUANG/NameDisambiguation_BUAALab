#coding: utf-8
__author__ = 'ssm'

import numpy as np
import datetime
import okm_local

from sklearn.manifold import TSNE
import matplotlib

def norm_vec(v):
    return (v+.0)/np.linalg.norm(v)

dwords = np.load("vecs.npy").item()
words = list(dwords.keys())
vecs = list(dwords.values())

centroids = np.load("afkmc2_init_vecs.npy")

k = len(centroids)

tagged, centroids, dist_to_centroid = okm_local.kmeans(vecs,k,centroids,iter_num=50)


for j in range(k):
    label_index = [i for i in range(len(tagged)) if tagged[i] == j]
    dists = [dist_to_centroid[i] for i in label_index]
    dists_sorted = sorted([(label_index[i], dists[i]) for i in range(len(dists))], key=lambda a: a[1],reverse=True)
    with open("./test_kmeans/kmeans_50_"+str(j)+".txt", "w") as f:
        a = (sum([d[1] for d in dists_sorted[:int(len(dists_sorted)/3)]]) + .0)/(len(dists_sorted)/3)
        f.write(str(a) + '\n')
        for d in dists_sorted[:]:
            f.write(str(words[d[0]]))
            f.write('\n')

np.save("./test_kmeans/kmeans_centroids_50", centroids)

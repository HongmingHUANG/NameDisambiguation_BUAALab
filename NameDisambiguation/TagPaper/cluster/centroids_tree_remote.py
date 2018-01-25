__author__ = 'ssm'
#Load the center of clusters.
#The result is like a tree.
#The center of a level-1 cluster and all centers of its level-2 clusters are one element of the entroids list.
import numpy as np

path = './skm_iter_70_2nd/iter_70_'

centroids = []

s1 = np.load("skm_iter_70_centroids.npy")
for i in range(14):
    s2 = np.load(path + str(i) + '/afk_centers.npy')
    centroids.append((s1[i], s2))
print(centroids)
#np.save("subject_tree", centroids)
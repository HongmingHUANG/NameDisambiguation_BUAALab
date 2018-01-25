__author__ = 'ssm'

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import datetime
from sklearn.cluster import MiniBatchKMeans, MeanShift, DBSCAN, Birch
import logging
import numpy as np
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
def sequential_kmeans(data, k, k_inits,iter_num=None, init_a=None):
    # assigned centroid index
    tagged_data = [0 for i in range(len(data))]
    # distance from assigned centroid last time
    data_last_dist = [-2 for i in range(len(data))]
    data_eu_last_dist = []
    if iter_num is None:
        iter_num = 10
    if init_a is None:
        init_a = 0.3
    centroids = k_inits
    mn = iter_num*len(data) # for update a
    flat_a = 0.0002  # flat learning rate
     # iter begin
    t = 0 # 0 < t < mn
    last_variance = 0
    end_count = 0
    for m in range(iter_num):
        variance = 0
        for n in range(len(data)):
            # get closest centroid
            last = data_last_dist[n]
            index = tagged_data[n]
            for i in range(len(centroids)):
                tmp = cosine_similarity([data[n]], [centroids[i]])[0][0]
                if tmp > last:
                    last = tmp
                    index = i
            tagged_data[n] = index
            data_last_dist[n] = last
            a = init_a * ( ((flat_a + .0)/init_a)**((t+.0)/mn) )
            centroids[index] = centroids[index] + a*(data[n]-centroids[index])
            t += 1
        #inner variance of cluster
        for di in range(len(data)):
            variance += euclidean_distances([centroids[tagged_data[di]]], [data[di]])[0][0]**2
        if (last_variance - variance < 0.000001) and (last_variance >= variance):
            end_count += 1
        else:
            end_count = 0
        if end_count == 4:
            print("end last iter {} at {} loss {}".format(m, datetime.datetime.now(), variance))
            break

        last_variance = variance
        print("end iter {} at {} loss {}".format(m, datetime.datetime.now(), variance))

    for i in range(len(data)):
        data_eu_last_dist.append(euclidean_distances([centroids[tagged_data[i]]], [data[i]])[0][0])

    return tagged_data, centroids, data_last_dist, data_eu_last_dist


def kmeans(data, k, k_inits,iter_num=None):
    # assigned centroid index
    tagged_data = [0 for i in range(len(data))]
    data_last_dist = [-2 for i in range(len(data))]
    data_eu_last_dist = []
    if iter_num is None:
        iter_num = 10
    centroids = k_inits

    # iter begin
    t = 0 # 0 < t < mn
    for m in range(iter_num):
        variance = 0
        for n in range(len(data)):
            # distance from assigned centroid last time
            # get closest centroid
            last = data_last_dist[n]
            index = tagged_data[n]
            for i in range(len(centroids)):
                tmp = cosine_similarity([data[n]], [centroids[i]])[0][0]
                if tmp > last:
                    last = tmp
                    index = i
            tagged_data[n] = index
            data_last_dist[n] = last

        for i in range(len(centroids)):
            this_set = [j for j in range(len(data)) if tagged_data[j] == i]
            sum_vec = 0
            for idx in this_set:
                sum_vec += data[idx]
            centroids[i] = sum_vec / (len(this_set) + .0)
        #inner variance of cluster
        for di in range(len(data)):
            variance += euclidean_distances([centroids[tagged_data[di]]], [data[di]])[0][0]**2
        print("end iter {} at {} loss {}".format(m, datetime.datetime.now(), variance))

    for i in range(len(data)):
        data_eu_last_dist.append(euclidean_distances([centroids[tagged_data[i]]], [data[i]])[0][0])

    return tagged_data, centroids, data_last_dist, data_eu_last_dist

def minibatchkmeans(data, k, k_inits):
    mbk_model = MiniBatchKMeans(n_clusters=k, init=k_inits, n_init=0)
    mbk = mbk_model.fit(data)
    dist_to_centroids = []
    eu_dist_to_centroids = []
    for i in range(len(mbk.labels_)):
        cent = mbk.cluster_centers_[mbk.labels_[i]]
        dist_to_centroids.append(cosine_similarity([cent], [data[i]])[0][0])
        eu_dist_to_centroids.append(euclidean_distances([cent], [data[i]])[0][0])
    return mbk.labels_, mbk.cluster_centers_, dist_to_centroids, eu_dist_to_centroids

def meanshift(data, k, k_inits, bandwidth, mb_freq):
    ms_model = MeanShift(bandwidth=bandwidth,seeds=k_inits, min_bin_freq=mb_freq)
    ms = ms_model.fit(data)
    dist_to_centroids = []
    eu_dist_to_centroids = []
    for i in range(len(ms.labels_)):
        cent = ms.cluster_centers_[ms.labels_[i]]
        dist_to_centroids.append(cosine_similarity([cent], [data[i]])[0][0])
        eu_dist_to_centroids.append(euclidean_distances([cent], [data[i]])[0][0])
    return ms.labels_, ms.cluster_centers_, dist_to_centroids, eu_dist_to_centroids

def dbscan(data):
    dbscan_model = DBSCAN()
    dbscan = dbscan_model.fit(data)
    k = len(set(dbscan.labels_))
    #mean of core samples as centroids
    #core_sample_dict = {label:[core_1_id, core_2_id,..], }
    core_sample_dict = {}
    for i in range(len(dbscan.core_sample_indices_)):
        index = dbscan.core_sample_indices_[i]
        label = dbscan.labels_[index]
        if label in core_sample_dict:
            core_sample_dict[label].append(index)
        else:
            core_sample_dict[label] = [index]
    for label_tag in core_sample_dict:
        core_list = core_sample_dict[label_tag]
        centroid = sum(core_list) / len(core_list)
        core_sample_dict[label_tag] = centroid
    #distance to centroid
    dist_to_centroids = []
    eu_dist_to_centroids = []
    for i in range(len(dbscan.labels_)):
        cent = core_sample_dict[dbscan.labels_[i]]
        dist_to_centroids.append(cosine_similarity([cent], [data[i]])[0][0])
        eu_dist_to_centroids.append(euclidean_distances([cent], [data[i]])[0][0])
    sorted_cent_dict = sorted(core_sample_dict.items(), key=lambda a: a[0])
    centroids = [a[1] for a in sorted_cent_dict]
    return dbscan.labels_, centroids, dist_to_centroids, eu_dist_to_centroids

def birch(data, k):
    birch_model = Birch(n_clusters=k)
    birch = birch_model.fit(data)
    #mean of each sample as centroids
    centroids = []
    for ki in range(k):
        ki_vecs = [data[i] for i in range(len(birch.labels_)) if birch.labels_[i] == ki]
        centroids.append(sum(ki_vecs)/ len(ki_vecs))
    #distance to centroid
    dist_to_centroids = []
    eu_dist_to_centroids = []
    for i in range(len(birch.labels_)):
        cent = centroids[birch.labels_[i]]
        dist_to_centroids.append(cosine_similarity([cent], [data[i]])[0][0])
        eu_dist_to_centroids.append(euclidean_distances([cent], [data[i]])[0][0])
    return birch.labels_, centroids, dist_to_centroids, eu_dist_to_centroids




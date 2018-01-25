#coding:utf-8
__author__ = 'ssm'

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys
import datetime
import clustering

'''
compare clustering algorithm:
kmeans, sequential kmeans, minibatchkmeans, meanshift, DBSCAN, BIRCH
calculate silhouette coefficient & my defined 'a' if there is
'''

def euclidean_distance(v1, v2):
    return euclidean_distances([v1],[v2])[0][0]

dwords = np.load("vecs.npy").item()
words = list(dwords.keys())
vecs = list(dwords.values())#[:int(len(words)/12)]
centroids = np.load("afkmc2_init_vecs.npy")
k = len(centroids)

time1 = datetime.datetime.now()
precomputed_distances = [[0 for i in range(len(vecs))] for j in range(len(vecs))]
for ci in range(len(vecs)):
    for cj in range(ci, len(vecs)):
        precomputed_distances[ci][cj] = euclidean_distance(vecs[ci], vecs[cj])
        precomputed_distances[cj][ci] = precomputed_distances[ci][cj]
time2 = datetime.datetime.now()
print("compute distances cost {}".format(time2-time1))

def calculate_s_score(index, tagged):
    current_cluster = [i for i in range(tagged) if tagged[i] == tagged[index]]
    sum_dists = 0
    for idx in current_cluster:
        if idx == index:
            continue
        sum_dists += precomputed_distances[index][idx]
    #a(label)
    a_o = sum_dists / (len(current_cluster)-1)
    #b(label)
    b_o = 9999999999999999999999999
    for idx in list(set(tagged)):
        if idx == tagged[index]:
            continue
        other_label_index = [i for i in range(len(tagged)) if tagged[i] == idx]
        other_index_sum = 0
        for iidx in other_label_index:
            other_index_sum += precomputed_distances[index][iidx]
        tmp = other_index_sum / len(other_label_index)
        if tmp < b_o:
            b_o = tmp
    #s
    s = (b_o - a_o + .0) / max(b_o, a_o)
    return s

def process_and_write(path, tagged, centroids, cosine_2_cent):
    '''
    :param path: write to path
    :param tagged: tagged data (labels)
    :param centroids: centroids vecs
    :param cosine_2_cent: cosine similarity of data[i] to its centroids
    :param eu_2_cent: eu distance of data[i] to its centroids
    :return:
    '''
    dir_path = './cmp_cluster/' + path + '/'

    for ki in range(len(centroids)):
        label_index = [i for i in range(len(tagged)) if tagged[i] == ki]
        cosine_dists = [cosine_2_cent[i] for i in label_index]
        #eu_dists = [eu_2_cent[i] for i in label_index]
        #===================my defined a===============
        dists_sorted = sorted([(label_index[i], cosine_dists[i]) for i in range(len(cosine_dists))]
                              , key=lambda a: a[1],reverse=True)
        a = (sum([d[1] for d in dists_sorted[:int(len(dists_sorted)/3)]]) + .0)/(len(dists_sorted)/3)

        # s = 0
        # count = 0
        # slist = []
        # for index,dist in dists_sorted[:-int(len(dists_sorted)/6)]:
        #     s += calculate_s_score(index, tagged)
        #     if count == [int(len(dists_sorted)/3), 2 * int(len(dists_sorted)/3)]:
        #         slist.append(s/count)
        # slist.append(s/count)

        #==============write to file========================
        with open(dir_path + str(ki) + ".txt", "w") as f:
            # f.write(str(slist) + '\t' + str(a) + '\n')
            f.write(str(a) + '\n')
            for d in dists_sorted[:]:
                #f.write(str(words[d[0]].encode('utf-8')))
                f.write(str(words[d[0]]))
                f.write('\n')
        print("end process centroid {}".format(ki))

def cmp_clustering():
    '''
    #k-means
    km_tagged, km_centroids, km_dist_2_cent, km_eu_d2c = clustering.kmeans(vecs,k,centroids,iter_num=30)
    print("end k-means at {}".format(datetime.datetime.now()))
    process_and_write("km", km_tagged, km_centroids, km_dist_2_cent, km_eu_d2c)
    '''

    #sequential k-means
    print("start at {}".format(datetime.datetime.now()))
    skm_tagged, skm_centroids, skm_dist_2_cent, skm_eu_d2c = clustering.sequential_kmeans(vecs,k,centroids,iter_num=50,
                                                                              init_a=0.008)
    print("end s-kmeans at {}".format(datetime.datetime.now()))
    process_and_write("skm", skm_tagged, skm_centroids, skm_dist_2_cent, skm_eu_d2c)


    # #minibatchkmeans
    # mbk_tagged, mbk_centroids, mbk_dist_2_cent = clustering.minibatchkmeans(vecs,k, centroids,
    #                                                                                     iter_num=1, batch_size=300)
    # print("end minibatchk-means at {}".format(datetime.datetime.now()))
    # process_and_write("mbk", mbk_tagged, mbk_centroids, mbk_dist_2_cent)

    '''
    #meanshift
    print("start meanshift at {}".format(datetime.datetime.now()))
    ms_tagged, ms_centroids, ms_dist_2_cent, ms_eu_d2c = clustering.meanshift(vecs,k,centroids,
                                                                              bandwidth=0.9,
                                                                              mb_freq=5)
    print("end meanshift at {}".format(datetime.datetime.now()))
    process_and_write("ms", ms_tagged, ms_centroids, ms_dist_2_cent, ms_eu_d2c)
    print("len centroids {}".format(len(ms_centroids)))


    #DBSCAN
    print("start dbscan at {}".format(datetime.datetime.now()))
    db_tagged, db_centroids, db_dist_2_cent, db_eu_d2c = clustering.dbscan(vecs)
    print("end dbscan at {}".format(datetime.datetime.now()))
    process_and_write("db", db_tagged, db_centroids, db_dist_2_cent, db_eu_d2c)


    #BIRCH
    birch_tagged, birch_centroids, birch_dist_2_cent, birch_eu_d2c = clustering.birch(vecs,k)
    print("end birch at {}".format(datetime.datetime.now()))
    process_and_write("birch", birch_tagged, birch_centroids, birch_dist_2_cent, birch_eu_d2c)
    '''
cmp_clustering()
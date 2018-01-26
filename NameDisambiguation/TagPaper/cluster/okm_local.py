__author__ = 'ssm'

from sklearn.metrics.pairwise import cosine_similarity
import datetime

def okm(data, k, k_inits, iter_num=None, init_a=None):
    '''
    :param data: 2d array (all unit vectors, using np.linalg.norm)
    :param k: k
    :param centroids: set init centroids (based on background knowledge)
    :param must_link: 2d array [[p1,p2],[p4,p3]..] p1,p2 must in same cluster
    :param mustn_link: 2d array must not in same cluster
    :param iter_num: default 10
    :param init_a: default 0.3
    :return: tagged_data, k centroids
    '''
    # assigned centroid index
    tagged_data = [0 for i in range(len(data))]
    # distance from assigned centroid last time
    data_last_dist = [-2 for i in range(len(data))]
    if iter_num is None:
        iter_num = 10
    if init_a is None:
        init_a = 0.3
    centroids = k_inits
    mn = iter_num*len(data) # for update a
    flat_a = 0.05  # flat learning rate

    # iter begin
    t = 0 # 0 < t < mn
    count_cent = [0 for i in range(k)] # points assigned to each centroid
    remote_k = [1 for i in range(k)] # the remotest k (avoid empty cluster)
    remote_k_index = [0 for i in range(k)]
    for m in range(iter_num):
        for n in range(len(data)):
            # get closest centroid
            last = data_last_dist[n]
            index = tagged_data[n]
            for i in range(len(centroids)):
                tmp = cosine_similarity([data[n]], [centroids[i]])
                if tmp > last:
                    last = tmp
                    index = i

            # add remote point and its index
            if last < max(remote_k):
                i = remote_k.index(max(remote_k))
                remote_k.pop(i)
                remote_k_index.pop(i)
                remote_k.insert(i,last)
                remote_k_index.insert(i, n)
            tagged_data[n] = index
            data_last_dist[n] = last
            count_cent[index] += 1

            # update learning rate (exponentially decrease)
            a = init_a * ( ((flat_a + .0)/init_a)**((t+.0)/mn) )

            # update centroid
            '''
            if cosine_similarity([data[n]], [centroids[index]]) > 0.75:
                    centroids[index] = centroids[index] + a*(data[n]-centroids[index])*0.2
            else:
                    centroids[index] = centroids[index] + a*(data[n]-centroids[index])

            if cosine_similarity([centroids[index]],[k_inits[index]]) < 0.65: # add constraints to centroid
                centroids[index] = centroids[index] + a*(k_inits[index]-centroids[index])
            '''
            centroids[index] = centroids[index] + a*(data[n]-centroids[index])

            t += 1
            if n % 100000 == 0:
                print('end {} {}'.format(n, datetime.datetime.now()))

        #avoid empty cluster
        i = count_cent.count(0)
        if i == 0:
            print("end iter {}".format(m))
            continue
        for i in range(k):
            if count_cent[i] == 0:
                if len(remote_k) != 0:
                    index = remote_k.index(min(remote_k)) # get remotest distance index
                    centroids[i] = data[remote_k_index[index]]
                    tagged_data[remote_k_index[index]] = i
                    data_last_dist[remote_k_index[index]] = 0.
                    remote_k.pop(index)
                    remote_k_index.pop(index)

    return tagged_data, centroids, data_last_dist

import collections

def violate_constraints(index_n, tag_n, must_link, mustn_link, tagged_data):
    '''
    check whether assigning index_n to tag_n violate constraints
    :param index_n: n
    :param tag_n: tag of data[index_n]
    :param must_link: c
    :param mustn_link: c
    :param tagged_data: all
    :return: 0-violate; 1-not violate and add weight; 2-do nothing;
    '''
    flag = False
    for a in must_link:
        if a[0] == index_n:
            flag = True
            if tag_n != tagged_data[a[1]]:
                return 0
        elif a[1] == index_n:
            flag = True
            if tag_n != tagged_data[a[0]]:
                return 0
    for b in mustn_link:
        if a[0] == index_n:
            flag = True
            if tag_n == tagged_data[a[1]]:
                return 0
        elif a[1] == index_n:
            flag = True
            if tag_n == tagged_data[a[0]]:
                return 0
    if flag:
        return 1
    else:
        return 2


def oskm_constrained(data, k, k_inits, must_link, mustn_link,
                     iter_num=None, init_a=None):
    '''
    :param data: 2d array (all unit vectors, using np.linalg.norm)
    :param k: k
    :param centroids: set init centroids (based on background knowledge)
    :param must_link: 2d array [[p1,p2],[p4,p3]..] p1,p2 must in same cluster
    :param mustn_link: 2d array must not in same cluster
    :param iter_num: default 10
    :param init_a: default 0.3
    :return: tagged_data, k centroids
    '''
    # assigned centroid index
    tagged_data = [0 for i in range(len(data))]
    # distance from assigned centroid last time
    data_last_dist = [-2 for i in range(len(data))]
    if iter_num is None:
        iter_num = 10
    if init_a is None:
        init_a = 0.3
    centroids = k_inits
    mn = iter_num*len(data) # for update a
    flat_a = 0.1  # flat learning rate

    # iter begin
    t = 0 # 0 < t < mn
    count_cent = [0 for i in range(k)] # points assigned to each centroid
    remote_k = [1 for i in range(k)] # the remotest k (avoid empty cluster)
    remote_k_index = [0 for i in range(k)]
    for m in range(iter_num):
        for n in range(len(data)):
            # get closest centroid
            last = data_last_dist[n]
            index = tagged_data[n]
            # store newly calculated distance from centroids (for constraint use)
            tmp_dis_dic = {}
            for i in range(len(centroids)):
                tmp = cosine_similarity([data[n]], [centroids[i]])[0][0]
                tmp_dis_dic[i] = tmp

            tmp_dis_dic = sorted(tmp_dis_dic.items(), key=lambda a: a[1], reverse=True)

            #constraint check, begin from 2nd iteration
            va = -1
            if m > 1:
                for dis in tmp_dis_dic:
                    va = violate_constraints(n, dis[0], must_link, mustn_link, tagged_data)
                    if va == 1 or va == 2:
                        last = dis[1]
                        index = dis[0]
                        break
            else:
                last = tmp_dis_dic[0][1]
                index = tmp_dis_dic[0][0]

            # add remote point and its index
            if last < max(remote_k):
                i = remote_k.index(max(remote_k))
                remote_k.pop(i)
                remote_k_index.pop(i)
                remote_k.insert(i,last)
                remote_k_index.insert(i, n)
            tagged_data[n] = index
            data_last_dist[n] = last
            count_cent[index] += 1

            # update learning rate (exponentially decrease)
            if va == 1:
                a = init_a * ( ((flat_a + .0)/init_a)**((t+.0)/mn) )
            else:
                a = init_a * ( ((flat_a + .0)/init_a)**((t+.0)/mn) )

            # update centroid
            '''
            if cosine_similarity([data[n]], [centroids[index]]) > 0.8:
                    centroids[index] = centroids[index] + a*(data[n]-centroids[index])
            else:
                    centroids[index] = centroids[index] + a*(data[n]-centroids[index])

            if cosine_similarity([centroids[index]],[k_inits[index]]) < 0.5: # add constraints to centroid
                centroids[index] = k_inits[index]
                '''
            centroids[index] = centroids[index] + a*(data[n]-centroids[index])

            t += 1
            if n % 100000 == 0:
                print('end {} {}'.format(n, datetime.datetime.now()))

        #avoid empty cluster
        for i in range(k):
            if count_cent[i] == 0:
                if len(remote_k) != 0:
                    index = remote_k.index(min(remote_k)) # get remotest distance index
                    centroids[i] = data[remote_k_index[index]]
                    tagged_data[remote_k_index[index]] = i
                    data_last_dist[remote_k_index[index]] = 0.
                    remote_k.pop(index)
                    remote_k_index.pop(index)

        print('end iter {}'.format(m))
    return tagged_data, centroids, data_last_dist

def sequential_kmeans(data, k, k_inits,iter_num=None, init_a=None):
    # assigned centroid index
    tagged_data = [0 for i in range(len(data))]
    # distance from assigned centroid last time
    data_last_dist = [-2 for i in range(len(data))]
    if iter_num is None:
        iter_num = 10
    if init_a is None:
        init_a = 0.3
    centroids = k_inits
    mn = iter_num*len(data) # for update a
    flat_a = 0.1  # flat learning rate
     # iter begin
    t = 0 # 0 < t < mn
    for m in range(iter_num):
        for n in range(len(data)):
            # get closest centroid
            last = data_last_dist[n]
            index = tagged_data[n]
            for i in range(len(centroids)):
                tmp = cosine_similarity([data[n]], [centroids[i]])
                if tmp > last:
                    last = tmp
                    index = i
            tagged_data[n] = index
            data_last_dist[n] = last
            a = init_a * ( ((flat_a + .0)/init_a)**((t+.0)/mn) )
            centroids[index] = centroids[index] + a*(data[n]-centroids[index])
            if n % 10000 == 0:
                print("end iter {} count {} at {}".format(m, n, datetime.datetime.now()))
        print("end iter {} at {}".format(m, datetime.datetime.now()))
    return tagged_data, centroids, data_last_dist


def kmeans(data, k, k_inits,iter_num=None):
    # assigned centroid index
    tagged_data = [0 for i in range(len(data))]
    data_last_dist = [-2 for i in range(len(data))]
    if iter_num is None:
        iter_num = 10
    centroids = k_inits

    # iter begin
    t = 0 # 0 < t < mn
    for m in range(iter_num):
        for n in range(len(data)):
            # distance from assigned centroid last time
            # get closest centroid
            last = data_last_dist[n]
            index = tagged_data[n]
            for i in range(len(centroids)):
                tmp = cosine_similarity([data[n]], [centroids[i]])
                if tmp > last:
                    last = tmp
                    index = i
            tagged_data[n] = index
            data_last_dist[n] = last
            if n % 10000 == 0:
                print("end iter {} count {} at {}".format(m, n, datetime.datetime.now()))

        for i in range(len(centroids)):
            this_set = [j for j in range(len(data)) if tagged_data[j] == i]
            sum_vec = 0
            for idx in this_set:
                sum_vec += data[idx]
            centroids[i] = sum_vec / (len(this_set) + .0)
        print("end iter {} at {}".format(m, datetime.datetime.now()))
    return tagged_data, centroids, data_last_dist
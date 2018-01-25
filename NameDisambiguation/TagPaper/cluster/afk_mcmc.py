__author__ = 'ssm'

'''
assumption free k-mcmc (afk_mc2)
reference: nips 2016 Fast and Provably Good Seedings for k-Means
https://las.inf.ethz.ch/files/bachem16fast.pdf

finding inits for keyword vecs (kejso.com)
'''

import random
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

#dist^2 of two vectors
def d2(c1, vec):
    #get distance
    return math.pow(euclidean_distances([c1], [vec]),2)
    #return math.pow(cosine_distances([c1], [vec]),2)
    #return np.linalg.norm([c1],[vec])


def cal_proposal_distribution(vecs, c1, n):
    dist = []
    for vec in vecs:
        dist.append(d2(c1, vec))
    sum_dist = sum(dist)
    q = list(map(lambda a: 0.5*(a+.0) / sum_dist + (1+.0)/(2*n), dist))
    return q

def construct_sample_list(q):
    q_ = list(map(lambda a:a*1e6, q))
    r = np.repeat(range(len(q)), q_)
    return list(r)

def d2_c(x, C):
    # C = {c1, c2, ...}
    # return min(d2(x, ct))
    tmp = map(lambda a: d2(x, a), C)
    return min(tmp)

def assumption_free_kmcmc(vecs, k, chainlen, cinit=None, cinit_index=None):
    '''
    :param vecs: data set
    :param k: k centroids
    :param chainlen: chain length
    :return: k inits
    '''
    #preprocessing calculate proposal distribution 'q'
    c1 = cinit
    c1i = cinit_index
    if c1i is None:
        c1i = random.randint(0, len(vecs))
        c1 = vecs[c1i]
    C = [c1,]
    Ci = [c1i,]
    q = cal_proposal_distribution(vecs, c1, len(vecs))
    sample_list = construct_sample_list(q) # index
    #find c2...ck
    for i in range(1,k):
        xi = random.sample(sample_list,1)[0]
        x = vecs[xi]
        dx = d2_c(x, C)
        for j in range(1, chainlen):
            yi = random.sample(sample_list,1)[0]
            y = vecs[yi]
            dy = d2_c(y, C)
            if (dy*q[xi])/(dx*q[yi]) > np.random.uniform():
                x = y
                xi = yi
                dx = dy
        C.append(x)
        Ci.append(xi)
    return C, Ci

def assumption_free_kmcmc_modified(vecs, k, chainlen, cinit=None, cinit_index=None):
    '''
    :param vecs: data set
    :param k: k centroids
    :param chainlen: chain length
    :return: k inits
    '''
    #preprocessing calculate proposal distribution 'q'
    c1 = cinit
    c1i = cinit_index
    if c1i is None:
        c1i = random.randint(0, len(vecs))
        c1 = vecs[c1i]
    C = [c1,]
    Ci = [c1i,]
    q = cal_proposal_distribution(vecs, c1, len(vecs))
    sample_list = construct_sample_list(q) # index
    #find c2...ck
    for i in range(1,k):
        xi = random.sample(sample_list,1)[0]
        x = vecs[xi]
        dx = d2_c(x, C)
        for j in range(1, chainlen):
            yi = random.sample(sample_list,1)[0]
            y = vecs[yi]
            dy = d2_c(y, C)
            if (dy*q[xi])/(dx*q[yi] + 1e-15) > np.random.uniform():
                x = y
                xi = yi
                dx = dy
        C.append(x)
        Ci.append(xi)
    return C, Ci



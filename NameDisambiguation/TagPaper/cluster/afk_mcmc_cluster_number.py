__author__ = 'ssm'

'''
a fast method of determining number of clusters(k) for large scale data set
based on asumption free kmcmc
'''

'''
input: data set, estimated number of clusters
output: number of clusters k

1. using afk mcmc select m points (m > k, k is the optimal cluster number)
2. randomly select one point(here I used m_points[0]), calculate euclidean distances & cosine distances,
   form a vec for each point[euclidean distances, cosine distances]
3. calculate pairwise distances of vecs and sort
4. find the abrupt slope, vecs before the slope indicate similar points
5. delete similar points and get final k
'''

import afk_mcmc as afkmc2
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances
from sklearn.cluster import AffinityPropagation, DBSCAN



def afkmc2_cluster_number_ap(data,  chainlen, m):
    '''
    :param data: 2d array
    :param chainlen: length of markov chain
    :param estimate_num: an estimated number, should be larger than optimal number
    :return:
    '''
    '''
    print("total data len : {}".format(len(data)))
    print("length of markov chain : {}".format(chainlen))
    print("initially picked : {}".format(m))
    '''
    #1
    m_points, m_index = afkmc2.assumption_free_kmcmc_modified(data, m, chainlen)


    # visualize
    #low_dim_embs = tsne.fit_transform(data[200:1000] + m_points)
    #low_dim_embs = low_dim_embs[800:]
    #plotvecs(low_dim_embs)

    #print [tags[i] for i in m_index]
    '''
    similarities = [[0 for i in range(m)] for j in range(m)]
    for ci in range(m):
        for cj in range(ci, m):
            similarities[ci][cj] = cosine_similarity([m_points[ci]],[m_points[cj]])[0][0]
            similarities[cj][ci] = similarities[ci][cj]
    pre = np.median(similarities)
    '''
    ap_model = AffinityPropagation()
    ap = ap_model.fit(m_points)

    '''
    #plot result
    colors = [(np.random.uniform(), np.random.uniform(), np.random.uniform())for i in range(len(ap.cluster_centers_indices_))]
    plt.figure(figsize=(6,6))
    for col_id in range(len(colors)):
        y_id = [i for i in range(len(ap.labels_)) if ap.labels_[i] == col_id]
        tmp_emb = [low_dim_embs[j] for j in y_id]
        x = [tmp_emb[k][0] for k in range(len(tmp_emb))]
        y = [tmp_emb[k][1] for k in range(len(tmp_emb))]
        plt.scatter(x, y, c = colors[col_id], marker = 'o')
    plt.show()
    '''
    return len(ap.cluster_centers_indices_), ap.labels_, m_index

def afkmc2_cluster_number_simple(data, tags,  chainlen, m):
    '''
    :param data: 2d array
    :param chainlen: length of markov chain
    :param estimate_num: an estimated number, should be larger than optimal number
    :return:
    '''
    print("total data len : {}".format(len(data)))
    print("length of markov chain : {}".format(chainlen))
    print("initially picked : {}".format(m))
    #1
    m_points, m_index = afkmc2.assumption_free_kmcmc_modified(data, m, chainlen)

    print([tags[i] for i in m_index])

    #2
    vecs = [[1, 0],]
    for i in range(1, m):
        #print m_points[0], m_points[i]
        vecs.append([cosine_distances([m_points[0]], [m_points[i]])[0][0],
                     euclidean_distances([m_points[0]], [m_points[i]])[0][0],
                     cosine_similarity([m_points[0]], [m_points[i]])[0][0]])
    #3
    pairwise_dist = {}
    for i in range(0, m-1):
        for j in range(i+1, m):
            pairwise_dist[(i, j)] = euclidean_distances([m_points[i]], [m_points[j]])[0][0]
    min_dist = min(pairwise_dist.values())
    for v in pairwise_dist:
        pairwise_dist[v] -= min_dist
    #4
    pairwise_dist = sorted(pairwise_dist.items(), key=lambda a: a[1])

    #print pairwise_dist
    '''
    for id, a in enumerate(pairwise_dist):
        print id, a
    '''

    max_diff = 0
    id = 0  # find abrupt slope id, vecs before id involve close points
    for i in range(1, len(pairwise_dist)):
        tmp =  pairwise_dist[i][1] - pairwise_dist[i-1][1]
        if tmp > max_diff:
            max_diff = tmp
            id = i
    print(id)
    #5
    m_id_list = [range(m)]
    for i in range(id):
        id1 = pairwise_dist[i][0][0]
        id2 = pairwise_dist[i][0][1]
        if id1 in m_id_list:
            m_id_list.remove(id1)

    return len(m_id_list)

def afkmc2_cluster_number_dbscan(data, tags,  chainlen, m):
    '''
    :param data: 2d array
    :param chainlen: length of markov chain
    :param estimate_num: an estimated number, should be larger than optimal number
    :return:
    '''
    '''
    print("total data len : {}".format(len(data)))
    print("length of markov chain : {}".format(chainlen))
    print("initially picked : {}".format(m))
    '''
    #1
    m_points, m_index = afkmc2.assumption_free_kmcmc_modified(data, m, chainlen)

    #print [tags[i] for i in m_index]
    db = DBSCAN(metric="euclidean", min_samples=2).fit(m_points)
    nclusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    return nclusters

'''
def plotvecs(low_dim_embs, filename='tsne.png'):
    plt.figure(figsize=(10, 10))  #in inches
    for i, label in enumerate(low_dim_embs):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)

    plt.show()
'''
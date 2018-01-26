__author__ = 'ssm'

'''
Draw plot pictures.
'''

from sklearn.manifold import TSNE
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

dwords = np.load("vecs2.npy").item()
words = list(dwords.keys())
vecs = list(dwords.values())

def plot_clusters(emb_clusters):
    cmap = cm.get_cmap('rainbow', 20)
    colors = cmap(np.linspace(0, 1, len(emb_clusters)))
    plt.figure(figsize=(18, 18))
    for i in range(len(emb_clusters)):
        for j in range(len(emb_clusters[i])):
            x, y= np.array(emb_clusters[i])[j,:]
            plt.scatter(x, y, c=colors[i])
    plt.savefig('./skm_iter_70_plot/__' +str(i) + '.png')

'''

path = './skm_iter_70_2nd/iter_70_'
import os
tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
import datetime
low_emb = tsne.fit_transform(vecs)
dwords = dict(zip(words, low_emb))
print('end at {}'.format(datetime.datetime.now()))

for i in range(14):
    files = os.listdir(path + str(i))
    clusters = []
    for j in range(len(files)-1):
        with open(path + str(i) + '/afk_' + str(j) + '.txt', "r") as f:
            lines = f.readlines()
            tmp = []
            for line in lines[1:]:
                tmp.append(dwords[line.strip()])
        clusters.append(low_emb)
    plot_clusters(clusters,i)

'''

path = './skm iter70/'
path2 = './cmp_cluster2/skm/'
import os
tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
import datetime
clusters = []
for i in range(14):
    with open(path2 + str(i) + '.txt', 'r', encoding='gbk') as f:
        lines = f.readlines()
        tmp = []
        for line in lines[1:900]:
            tmp.append(dwords[line.strip()])
        print(np.array(tmp).shape)
    clusters.append(tmp)
    print(clusters[i][0].shape)
all_vecs = []
for i in range(14):
    all_vecs += clusters[i]
print('all_vecs', np.array(all_vecs).shape)
low_emb = tsne.fit_transform(all_vecs)
print('end tsne')
low_emb_clusters = []
pos = 0
for i in range(14):
    low_emb_clusters.append(low_emb[pos:pos + len(clusters[i])])
    pos += len(clusters[i])
print('low_emb_clusters', np.array(low_emb_clusters).shape)
plot_clusters(low_emb_clusters)


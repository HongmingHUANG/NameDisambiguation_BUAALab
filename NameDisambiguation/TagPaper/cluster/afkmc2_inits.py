__author__ = 'ssm'

import numpy as np
import afk_mcmc as amc

dist = [1, 2, 3]
sum_dist = sum(dist)
wv = np.load("vecs.npy").item()
words = list(wv.keys())
vecs = list(wv.values())
print(len(words))
print(len(vecs))

k = 14
m = 20

cinit = wv[words[31]]
cinit_index = words.index(words[31])
C, Ci = amc.assumption_free_kmcmc(vecs, k, m, cinit=cinit, cinit_index=cinit_index)

for i in Ci:
    print(words[i])

np.save("akfmc2_init_vecs2",C)

# init = np.load("afkmc2_init_vecs.npy")
# print(init.shape)
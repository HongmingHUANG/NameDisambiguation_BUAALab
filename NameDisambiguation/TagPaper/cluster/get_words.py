import redis

__author__ = 'ssm'

'''
randomly get 120k word+wordvec from redis
Save these words' vector into vecs2.npy
The redis server contains all words and their vectors.
'''
import random
import numpy as np
import datetime

#This is the redis server. But which server?
r = redis.StrictRedis(host='10.2.4.78', port=6379, db=1)
wv = {}
with open("all_keywords.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()
    count = 0
    i = 0
    ri = random.sample(range(len(lines)), 150000)
    while count != 120000 and i < 150000:
        # get random number
        tmp_w = lines[ri[i]].strip('\n')
        i += 1
        # if word already exist, continue
        vec = r.get(tmp_w)
        if vec is None:
            #print("{} not in r".format(tmp_w))
            #exit()

            continue
        vec = vec.decode("utf-8")
        vec = vec.split(' ')
        vec = [float(w) for w in vec]
        wv[tmp_w] = np.array(vec, dtype=np.float32)
        count += 1
        if count % 1000 == 0:
            print("end {} {}".format(count, datetime.datetime.now()))
print(len(wv.keys()))
np.save("vecs2", wv)

# dwords = np.load("vecs.npy").item()
# words = list(dwords.keys())
# vecs = list(dwords.values())
# print(vecs[:1])
#
# dwords = np.load("vecs2.npy").item()
# words = list(dwords.keys())
# vecs = list(dwords.values())
# print(vecs[:1])
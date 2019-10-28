import pickle
import numpy as np
import sys
def eigen(num, layer_num, dataset):
    layer_num = int(layer_num)
    num = str(num)
    cur = pickle.load(open('config/{}_{}.pkl'.format(dataset, num), 'rb'))
    print(cur)
    curt = []
    layer_num = len(cur)
    for i in cur:
        if i != 'M':
            curt.append(i)
    Delta = []
    split_index = {}
    vector = {}
    #GP = [[0,1], [2,3], [4,5,6,7,8,9,10,11,12,13]]
    GP = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
    for group in range(len(GP)):
        for i in GP[group]:
            w = pickle.load(open('eigen/{}_A_{}_{}_.pkl'.format(dataset, str(i), num), 'rb'), encoding='latin1')
            if i == 0:
                W = w
            else:
                W = np.concatenate([W, w], 0)
        st = np.argsort(W)
        grow_rate = 0.3
        t = int(grow_rate * W.shape[0])
        thre = W[st[t]]
        delta = []
        for i in GP[group]:
            if i == 0:
                k = 3
            else:
                k = 1
            w = pickle.load(open('eigen/{}_A_{}_{}_.pkl'.format(dataset, str(i), num), 'rb'), encoding='latin1')
            v = pickle.load(open('eigen/{}_V_{}_{}_.pkl'.format(dataset, str(i), num), 'rb'), encoding='latin1')
            index = np.argwhere((w) < thre)
            l = index.shape[0]
            split_index[i] = np.squeeze(index)
            vector[i] = np.reshape(v, [v.shape[0], -1, k, k])
            delta.append(l)
        Delta += delta

    pickle.dump(split_index, open('eigen/{}_min.pkl'.format(num), 'wb'))
    pickle.dump(vector, open('eigen/{}_minv.pkl'.format(num), 'wb'))
    Delta = np.array(Delta)
    pickle.dump(Delta, open('config/delta_{}_{}.pkl'.format(dataset, str(int(num) + 1)), 'wb'))


eigen(sys.argv[1], sys.argv[2], sys.argv[3])

import pickle
import numpy as np
import sys


def eigen(num, split_num, layer_num):
    prefix = 'min_'
    layer_num = int(layer_num)
    num = str(num)
    #cur = [8, 8, 8, 8, 16, 16, 24, 24, 24, 24, 24, 24, 32, 32]
    #cur = [10, 12, 13, 13, 21, 29, 35, 37, 35, 25, 28, 28, 37, 32]
    #cur = [12, 12, 18, 17, 28, 54, 55, 45, 40, 25, 28, 28, 37, 32]
    #cur = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    cur = [22, 23, (20, 2), 25, (22, 2), 25, (24, 2), 20, 18, 19, 19, 20, (18, 2), 20]
    cur = [27, 39, (22, 2), 39, (37, 2), 40, (30, 2), 20, 18, 21, 21, 21, (19, 2), 20]
    cur = [29, 74, (24, 2), 54, (50, 2), 64, (42, 2), 21, 18, 24, 21, 21, (19, 2), 20]
    cur = [33, 132, (28, 2), 69, (59, 2), 104, (53, 2), 21, 18, 24, 21, 21, (19, 2), 20]
    cur = [33, 209, (34, 2), 90, (72, 2), 160, (64, 2), 21, 18, 24, 21, 21, (19, 2), 20]
    cur[2] = cur[2][0]
    cur[4] = cur[4][0]
    cur[6] = cur[6][0]
    cur[12] = cur[12][0]
    cur = [4,4,4,4]
    cur = [4, 7, 5, 4]
    cur = [10, 12, 21, 11]
    cur = [11, 18, 29, 12]
    cur = [11, 18, 29, 12]
    cur = [11, 30, 38, 12]
    print(cur)
    cur = pickle.load(open('cifar10max' + str(num) + '.pkl', 'rb'))
    curt = []
    DD = 0
    layer_num = len(cur)
    '''
    for i in cur:
        if i != 'M':
            curt.append(i)
    for i in range(layer_num):
        #w = pickle.load(open('eigen/' + prefix+ 'A_' + str(i) + '_' + num + '_.pkl', 'rb'), encoding='latin1')
        try:
            w = pickle.load(open('eigenm/' + prefix+ 'A_' + str(i) + '_' + num + '_.pkl', 'rb'), encoding='latin1')
            #w1 = pickle.load(open('eigen/' + prefix+ 'A_' + str(i) + '_' + num + '_.pkl', 'rb'), encoding='latin1')
            print(w)
            #print(w.shape)
        except:
            DD = DD + 1
            continue
        if i == DD:
            W = w
        else:
            W = np.concatenate([W, w], 0)
    '''
    prefix = 'max_'
    r = [0.116849326, 0.038422294, 0.02061177, 0.02997986, 0.014377874, 0.0062844744, 0.012592447, 0.006363712, 0.008475702, 0.02377023, 0.038945824, 0.03370137, 0.03196905, 0.06754288]
    r = np.ones([14])
    #r = pickle.load(open('cifar10max' + str(num) + 'mag.pkl','rb'))
    for i in range(layer_num):
        #w = pickle.load(open('eigen/' + prefix+ 'A_' + str(i) + '_' + num + '_.pkl', 'rb'), encoding='latin1')
        try:
            w = pickle.load(open('eigenmax/' + prefix+ 'A_' + str(i) + '_' + num + '_.pkl', 'rb'), encoding='latin1')
            #print(np.mean(w))
            w *= np.sqrt(r[i])
            print(np.mean(w))
            #w1 = pickle.load(open('eigen/' + prefix+ 'A_' + str(i) + '_' + num + '_.pkl', 'rb'), encoding='latin1')
            #print(w.shape)
        except:
            DD = DD + 1
            continue
        if i == DD:
            W = w
        else:
            W = np.concatenate([W, -w], 0)
    st = np.argsort(W)
    L = W.shape[0]
    t = int(0.15 * L)
    thre = W[st[t]]
    SP = {}
    VP = {}
    SP1 = {}
    VP1 = {}
    DL = []
    dp = []
    prefix = sys.argv[3] + '_'
    for i in range(layer_num):
        if i == 0:
            k = 3
        else:
            k = 1
        try:
            w = pickle.load(open('eigenmax/' +prefix+ 'A_' + str(i) + '_' + num + '_.pkl', 'rb'), encoding='latin1')
            v = pickle.load(open('eigenmax/' +prefix+ 'V_' + str(i) + '_' + num + '_.pkl', 'rb'), encoding='latin1')
            w *= np.sqrt(r[i])
        except:
            print(i)
            l = int(0.1 * curt[i])
            D = np.random.randint(0, curt[i], size=[int(0.1 * curt[i]), 1])
            SP[i] = D
            VD = np.zeros([1, 1, 1])
            #VP[i] = np.reshape(v, [v.shape[0], -1, k, k])
            VP[i] = np.zeros([curt[i], curt[i-1], 1, 1])
            DL.append(l)
            continue
        if prefix == 'max_':
            ic = -1
        else:
            ic = 1
        D = np.argwhere((ic * w) < thre)
        l = D.shape[0]
            
        SP[i] = np.squeeze(D)
        #SP1[i] = np.random.randint(0, curt[i], size=[D.shape[0], 1])
        VD = v[D].astype(float)
        VP[i] = np.reshape(v, [v.shape[0], -1, k, k])
        #VP1[i] = np.zeros_like(VD)
        dp.append(l)
        DL.append(l)
        print(SP[i].shape)
        print(VP[i].shape)
        print(cur[i])
   
    pickle.dump(SP, open('eigenmax/' + num + prefix + 'global.pkl', 'wb'))
    pickle.dump(VP, open('eigenmax/' + num + prefix + 'globalv.pkl', 'wb'))
    print(DL)
    DL = np.array(DL)
    ct = 0
    DDL = []
    
    for i in cur:
        if i == 'M':
            DDL.append('M')
            continue
        else:
            
            DDL.append(int(i + DL[ct]))
            ct += 1
    
    for i in range(len(cur)):
        cur[i] = int(DDL[i])
    #print(DL)
    print(DDL)
    print(cur)
    pickle.dump(DL, open('maxcfg' + str(int(num) + 1) + '.pkl', 'wb'))
    #print(SP)


eigen(sys.argv[1], 1, sys.argv[2])

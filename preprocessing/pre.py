from sklearn.neural_network import BernoulliRBM

from sklearn import preprocessing as p
from sklearn.decomposition import PCA
import numpy as np

def get_processor( args):

    proc_type = args['type']
    print (proc_type)
    # params = args['params']
    if proc_type =='standard': # 0 mean , 1 variance
        proc =  p.StandardScaler()

    elif proc_type =='normalize': #  1 norm
        proc = p.Normalizer()
    elif 'scale': # 0:1 scale
         proc = p.MinMaxScaler()
    elif proc_type =='pca': # pca with n_compnents = min(n_samples, n_features)
        proc = PCA()
    elif proc_type =='rbm':
        proc = BernoulliRBM(random_state=0, verbose=True)
    elif proc_type =='log': # to be implemented
        proc = None #TODO: implement log scaling
    else:
        proc = None

    return proc

def remove_outliers(y):
    # print min(y), max(y), np.mean(y)
    m = np.mean(y)
    s = np.std(y)
    print min(y), max(y), np.mean(y)
    print 's', s
    y = y-m
    s = np.std(y)
    y[y>2*s] = 2*s
    y[y<-2*s] = -2*s
    print min(y), max(y), np.mean(y)
    return y

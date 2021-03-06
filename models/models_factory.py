from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import additive_chi2_kernel

def get_model(params):
    model_type = params['type']
    p = params['params']
    print ('model type: ', model_type)
    print ('model paramters: ', p)

    if model_type =='svm':
        if p['kernel'] == 'additive_chi2_kernel':
            p['kernel'] = additive_chi2_kernel
        model =  SVC(max_iter=1000 , **p)

    elif model_type == 'decisionTree':
        model = DecisionTreeClassifier(**p)
    elif model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(**p)
    elif model_type == 'MLPClassifier':
        model = MLPClassifier(**p)

    else:
        print ('wrong model type, default model will be used')
        model =  SVR(C=0.1, epsilon=0.4, gamma=0.1)

    if 'bag' in params:
        bag_params = params['bag']
        n_estimators = bag_params['n_estimators']
        n_jobs = bag_params['n_jobs']
        print ('bagging: n_estimators, n_jobs: ', n_estimators, n_jobs)
        print ('WARNING: bagging takes long time')
        model = BaggingRegressor(model, n_estimators=n_estimators, n_jobs=n_jobs)

    return model

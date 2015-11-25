import logging
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_model(params):
    model_type = params['type']
    p = params['params']
    logging.info('model type: %s', model_type)
    logging.info('model paramters: %s', p)

    if model_type =='svm':
        model =  SVC(max_iter=1000 , **p)
    # elif model_type =='ridge':
    #     model = Ridge(**p)
    #
    # elif model_type =='elastic':
    #     model = ElasticNet(**p)
    #
    # elif model_type =='lasso':
    #     model = Lasso(**p)
    elif model_type == 'decisionTree':
        model = DecisionTreeClassifier(**p)
    elif model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(**p)
    elif model_type == 'MLPClassifier':
        model = MLPClassifier(**p)
    else:
        logging.info('wrong model type, default model will be used')
        model =  SVR(C=0.1, epsilon=0.4, gamma=0.1)

    if 'bag' in params:
        bag_params = params['bag']
        n_estimators = bag_params['n_estimators']
        n_jobs = bag_params['n_jobs']
        logging.info('bagging: n_estimators, n_jobs: ', n_estimators, n_jobs)
        logging.info('WARNING: bagging takes long time')
        model = BaggingRegressor(model, n_estimators=n_estimators, n_jobs=n_jobs)

    return model

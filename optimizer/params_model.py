from hyperopt import hp
import numpy as np
from sklearn.metrics.pairwise import additive_chi2_kernel


#---------------------- SVM -----------------------
# http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
# http://www.svms.org/parameters/
# http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
C = hp.choice('svm_C',  np.logspace(-4, 6, 50))
gamma =  hp.choice('svm_gamma',  np.logspace(-9, 4, 50))
degree =  hp.choice('poly_degree', range(3,4))

# gamma =  10
# degree =  3

svm = {
        'type': 'svm',
        'params': hp.pchoice('svm_model_choice',
                            [ (0.2, {'C': C, 'kernel': 'linear', 'gamma':gamma, 'degree':1 }),
                              (0.2, {'C': C, 'kernel': 'rbf', 'gamma':gamma, 'degree':1}),
                              (0.6, {'C': C, 'kernel': 'poly', 'gamma':gamma, 'degree':degree})]
                            ),
    }


#-------------------decisionTree ----------------------------
features_perct = hp.uniform('max_uniform', 0, 1)
decisionTree = {
        'type': 'decisionTree',
        'params':{
            'max_depth': hp.choice('max_depth', range(2,30)),
            'max_features': 'auto', #hp.choice('max_features', [features_perct, 'log2', 'auto']),
            # 'splitter': hp.choice('splitter', ['best', 'random'])
            # 'splitter': hp.choice('splitter',  'random')
            'splitter': 'best'
        }
    }

#-------------------RandomForestRegressor --------------------------
randomForestClassifier = {
        'type': 'RandomForestClassifier',
        'params':{
            'max_depth': hp.choice('max_depth2', range(5,40)),
            'max_features': 'auto', #hp.choice('max_features2', [features_perct, 'log2', 'auto']),
            'n_estimators': hp.choice('forst_estimators', [10,20,30,40,50])
        }
}

#-------------------Neural Networks --------------------------
algorithm = hp.choice('MLP_algorithm', ['l-bfgs', 'sgd', 'adam'])
alpha = hp.choice('MLP_alpha',  np.logspace(-5, 2, 50))
hidden_layer_sizes = (hp.choice('MLP_depth1', range(3,30)), hp.choice('MLP_depth2', range(3,30)))
random_state = 1
batch_size = 500

MLPClassifier = {
                'type': 'MLPClassifier',
                'params': { 'algorithm': algorithm,
                            'alpha': alpha,
                            'hidden_layer_sizes':hidden_layer_sizes,
                            'random_state':random_state,
                            'batch_size':batch_size
                }
            }

#---------------------- choose models ---------------

# m=[svm, decisionTree, randomForestClassifier, MLPClassifier]
m=[svm]
# m=[decisionTree]
# m=[randomForestClassifier]
# m=[MLPClassifier]

choices = hp.choice('model', m)

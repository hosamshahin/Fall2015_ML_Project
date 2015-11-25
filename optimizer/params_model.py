from hyperopt import hp
import numpy as np
#---------------------- svm ---------------
# http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
# http://www.svms.org/parameters/
# http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf



#---------------------- SVM -----------------------
C = hp.choice('svm_C',  np.logspace(-2, 10, 50))
gamma =  hp.choice('svm_gamma',  np.logspace(-9, 3, 50))
svm = {
        'type': 'svm',
        'params': hp.pchoice('p',
                            [ (0.2, {'C': C, 'kernel': 'linear', }),
                              (0.8, {'C': C, 'kernel': 'rbf', 'gamma':gamma})]
                             ),
    }

#---------------------- ridge ---------------

# alpha = hp.uniform('alpha', 0, 1)
# ridge ={
#         'type': 'ridge',
#         'params':{'alpha': alpha},
#         'bag': {'n_estimators': hp.choice('n_estimators_ridge',[1, 5]), 'n_jobs': 1
#                 }
#     }
#
#
# #---------------------- elastic ---------------
# elastic_alpha = hp.uniform('elastic_alpha', 0, 1)
# l1_ratio = hp.uniform('l1_ratio', 0, 1)
# elastic = {
#      'type': 'elastic',
#      'params': {'l1_ratio': l1_ratio, 'alpha': elastic_alpha}
#      }
#
# #---------------------- lasso ---------------
#
# lasso_alpha = hp.uniform('lasso_alpha', 0, 1)
# lasso = {
#      'type': 'lasso',
#      'params': {'alpha': lasso_alpha}
#      }

#-------------------decisionTree ----------------------------

features_perct = hp.uniform('max_uniform', 0, 1)
decisionTree = {
        'type': 'decisionTree',
        'params':{
            'max_depth': hp.choice('max_depth', range(1,20)),
            'max_features': 'auto',
            'splitter': hp.choice('splitter', ['best', 'random'])
        },
        'bag': {
                'n_estimators': hp.choice('n_estimators_tree',[10, 20]),
                 'n_jobs': 1

        }
    }

#-------------------RandomForestClassifier --------------------------
randomForestClassifier = {
                'type': 'RandomForestClassifier',
                'params':{
                    'max_depth': hp.choice('max_depth2', range(1,20)),
                    'max_features': 'auto'
                }
            }

#-------------------Neural Networks --------------------------
algorithm = hp.choice('MLP_algorithm', ['l-bfgs', 'sgd', 'adam'])
alpha = hp.choice('MLP_alpha',  np.logspace(-5, 3, 50))
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

m=[svm, decisionTree, randomForestClassifier, MLPClassifier]

choices = hp.choice('model', m)

import logging, sys, yaml
from train_predict import run

# setup logging
logging.basicConfig(filename='../logs/run_model.log',
                    filemode='w',
                    format='%(asctime)s - {%(filename)s:%(lineno)d} - %(message)s',
                    datefmt='%m/%d %I:%M',
                    level=logging.DEBUG) # or logging.DEBUG
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


model_params = {
                'type': 'svm',
                        'params':
                            {
                                'C': 1,
                                'gamma': 5.68,
                                'kernel': 'rbf'
                            }
                }
data_params =  {
                'p1': False,
                'p2': 'sum'
                }

pre_params =    {
                'type': None
                }


# args = {}
# args['data_params'] = data_params
# args['model_params'] = model_params
# args['pre_params'] = pre_params
# logging.info("Args: %s", args)

# args = {'model_params': {'params': {'max_features': 'auto', 'max_depth': 16}, 'type': 'RandomForestClassifier'}, 'data_params': 'p2', 'pre_params': {'type': None}}
args = {'model_params': {'params': {'alpha': 1.0000000000000001e-05, 'random_state': 1, 'algorithm': 'l-bfgs', 'batch_size': 500, 'hidden_layer_sizes': (16, 26)}, 'type': 'MLPClassifier'}, 'data_params': None, 'pre_params': {'type': 'pca'}}

score = run(args, vis=True, save_vis=True)
logging.info("Final Score: %s", score)



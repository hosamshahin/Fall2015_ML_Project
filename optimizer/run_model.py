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

args = {}
args['data_params'] = data_params
args['model_params'] = model_params
args['pre_params'] = pre_params
logging.info("Args: %s", args)

args = {'model_params': {'params': {'max_features': 'auto', 'max_depth': 3}, 'type': 'RandomForestClassifier'}, 'data_params': None, 'pre_params': {'type': 'normalize'}}

score = run(args, vis=True, save_vis=True)
logging.info("Final Score: %s", score)

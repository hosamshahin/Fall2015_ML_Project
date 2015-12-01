import logging, sys, yaml, json
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

args = {'model_params': {'params': {'kernel': 'poly', 'C': 0.020235896477251575, 'degree': 4, 'gamma': 6.2505519252739763}, 'type': 'svm'}, 'data_params': None, 'pre_params': {'type': 'normalize'}}

tests = {}
with open('../results/run_hyperOpt.json', 'r') as f:
     json.load(f, tests)

print tests


# score = run(args, vis=False, save_vis=False, save_model=True)
# logging.info("Final Score: %s", score)
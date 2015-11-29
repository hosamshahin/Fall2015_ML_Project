import hyperopt
import params_data
import params_model
import numpy as np
import logging, sys, datetime
import yaml, json

from hyperopt import fmin, tpe, Trials
from train_predict import run
from matplotlib import pyplot as plt


# setup logging
logging.basicConfig(filename='../logs/run_hyperopt.log',
                    filemode='w',
                    format='%(asctime)s - {%(filename)s:%(lineno)d} - %(message)s',
                    datefmt='%m/%d %I:%M',
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("Start runner_auto")

# prepare hyperopt search space
space = {'data_params': params_data.data_choices,
         'pre_params': params_data.prep_choices,
         'model_params': params_model.choices}

trials = Trials()

best = fmin(run, space, algo=tpe.suggest, max_evals=300, trials=trials)


logging.info("~~~~~~~~~~~~~~~~~ fmin Done ~~~~~~~~~~~~~~~~~")
logging.info("Best Model:")
bestParams = hyperopt.space_eval(space, best)
logging.info(bestParams)
min = np.min(trials.results)
logging.info("Best loss is: %s", min['loss'])

# Save output
timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())

tests = list()
logging.info("Objective Func values for all trials: ")
for trial in trials.trials:
    rval = {}
    for k, v in trial['misc']['vals'].items():
        if v:
            rval[k] = v[0]

    test = hyperopt.space_eval(space, rval)
    test['result'] = trial['result']['loss']
    tests.append(test)
    logging.info(test['result'])


#~~~~ Writing Yaml data
# with open('../results/hyperOpt_svm'+'.yaml', 'w') as f:
     # yaml.dump(tests, f)
#~~~~ Writing json data
with open('../results/hyperOpt_svm'+'.json', 'w') as f:
     json.dump(tests, f)

# Reading data back
# with open('../results/trials2fixed'+timeStamp+'.yaml', 'r') as f:
#      tests = yaml.load(f)

# print(tests)
#~~~ Run one of the tests again
# run(tests[0])


#~~~~ Plots
# ids = [t['tid'] for t in trials.trials]
# cases = [t['misc']['vals']['data']['scaling_factor'] for t in trials.trials] # aggr is the name of the choice
# plt.xlabel("Trial index")
# plt.ylabel('Parameter aggr (methyl aggregation method)')
# plt.title("Values used during random search")
# plt.scatter(ids, cases)
#
# fvals = [t['result']['loss'] for t in trials.trials]
# scaling_factor = [t['misc']['vals']['data']['scaling_factor'] for t in trials.trials]
# plt.xlabel("Trial index")
# plt.ylabel('Parameter: scaling_factor')
# plt.title("Effect of scaling_factor druring random search")
# plt.scatter(scaling_factor, fvals)

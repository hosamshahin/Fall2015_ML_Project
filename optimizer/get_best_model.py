import logging, sys, yaml, json

from train_predict import run

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key):byteify(value) for key,value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

# setup logging
logging.basicConfig(filename='../logs/run_model.log',
                    filemode='w',
                    format='%(asctime)s - {%(filename)s:%(lineno)d} - %(message)s',
                    datefmt='%m/%d %I:%M',
                    level=logging.DEBUG) # or logging.DEBUG
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


with open('../results/trials/run_hyperopt_svm_5.json') as data_file:
    trials = json.load(data_file)

trials = byteify(trials)

bestResults = {}
bestModels = {}
for model in trials:
    model_type = model['model_params']['type']
    model_result = model['error']
    if model_type in bestResults:
        if model_result < bestResults[model_type]:
            bestResults[model_type] = model_result
            bestModels[model_type] = model
    else:
        bestResults[model_type] = model_result
        bestModels[model_type] = model

# save model
for key, value in bestModels.iteritems():
    score = run(value, vis=False, save_vis=False, save_model=True, save_cm=True)

# args = {'model_params': {'params': {'kernel': 'poly', 'C': 0.020235896477251575, 'degree': 4, 'gamma': 6.2505519252739763}, 'type': 'svm'}, 'data_params': None, 'pre_params': {'type': 'normalize'}}
# score = run(args, vis=False, save_vis=False, save_model=True)
# logging.info("Final Score: %s", score)

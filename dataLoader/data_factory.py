import numpy as np
import yaml
import os

# Load config
def get_config():
    print(os.getcwd())
    f = open('../config.yaml')
    config = yaml.safe_load(f)
    f.close()
    return config

config = get_config()
data_folder = config['data']['path']


def get_data(params, phase='train'):

    if phase == 'train':
        X = np.load(config['data']['training_data'])
        y = np.load(config['data']['training_labels'])
    elif phase == 'test':
        X = np.load(config['data']['testing_data'])
        y = np.load(config['data']['testing_labels'])

    y = y.reshape(-1)
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    return X, y

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


def get_data(params):

    X = np.load(config['data']['training_data'])
    print("X shape: ", X.shape)
    y = np.load(config['data']['training_labels'])
    y = y.reshape(-1)
    print("y shape: ", y.shape)

    return X, y

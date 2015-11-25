import numpy as np
import yaml
import os
import logging

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load config
def get_config():
    logging.info(os.getcwd())
    f = open('../config.yaml')
    config = yaml.safe_load(f)
    f.close()
    return config

config = get_config()
data_folder = config['data']['path']


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_data(params):

    # n_samples, n_features = 100, 100
    # X = np.random.randn(n_samples, n_features)
    # y = np.random.randn(n_samples)

    X = np.load(config['data']['training_data'])
    logging.info("X shape: %s", X.shape)
    y = np.load(config['data']['training_labels'])
    y = y.reshape(-1)
    logging.info("y shape: %s", y.shape)

    return X, y

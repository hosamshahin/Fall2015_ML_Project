import logging, sys, os
import cPickle as pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, hamming_loss
from matplotlib import pyplot as plt
from models.models_factory import get_model
from dataLoader.data_factory import *
from preprocessing import pre
from sklearn.metrics.pairwise import additive_chi2_kernel


def visualize(data, scores, args, save_vis =False):

    model_params = args['model_params']
    score_corr_train , score_corr_test = scores
    y_train, y_test, pred_train, pred_test = data

    traning_parameters = 'running information: ' + model_params['type'] + str(model_params['params'])
    plt.suptitle(traning_parameters, fontsize=11)

    x_range = np.arange(y_test.shape[0])
    print(x_range)
    print(y_test.dtype)

    plt.subplot(2, 1, 1)
    # plt.plot(y_test, 'b')
    # plt.plot(pred_test_value, 'r')
    plt.scatter(x=x_range, y=y_test, c='b', marker='x', linewidths=5)
    plt.scatter(x=x_range, y=pred_test, c='r')
    plt.legend(['truth', 'prediction'])
    plt.title('Testing performance' + '  testing correlation score: ' + str(score_corr_test), fontsize=8)
    # plt.show()

    x_range2 = np.arange(y_train.shape[0])
    plt.subplot(2, 1, 2)
    plt.scatter(x=x_range2, y=y_train, c='b', marker='x', linewidths=5)
    plt.scatter(x=x_range2, y=pred_train, c='r')
    plt.legend(['truth', 'prediction'])
    plt.title('Training performance' + '  training correlation score: ' + str(score_corr_train), fontsize=8)

    if save_vis:
        plt.savefig('../results/' + traning_parameters + '.png')

    plt.show()
    # plt.close()


def run(args, vis=False, save_vis=False, save_model=False):
    logging.info("Running new experiment\n========================\n")

    data_params = args['data_params']
    model_params = args['model_params']
    pre_params = args['pre_params']
    logging.info(args)

    # get data
    X, y = get_data(data_params)

    logging.info('y shape: %s', y.shape)
    logging.info('x shape: %s', X.shape)
    # get model
    model = get_model(model_params)

    if save_model:
        model_type = args['model_params']['type']
        with open('../results/' + model_type + '.pkl', 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # preprocessing
    proc = pre.get_processor(pre_params)

    if proc:
        proc.fit(X_train)
        X_train = proc.transform(X_train)
        X_test = proc.transform(X_test)
    else:
        print('no preprocessing applied')


    logging.info('fitting model started ....')
    model.fit(X_train, y_train)
    logging.info('model fitting finished')

    pred_test = model.predict(X_test)
    pred_train = model.predict(X_train)

    score_train = hamming_loss(y_train, np.int32(pred_train))
    score_test = hamming_loss(y_test, np.int32(pred_test))


    logging.info('score_test: %s', score_test)
    logging.info('score_train: %s', score_train)

    if vis:
        scores = [score_train , score_test]
        data = [y_train, y_test, pred_train, pred_test]
        visualize(data, scores, args, save_vis=save_vis)

    return score_test

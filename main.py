import sys
import os
import cv2
import numpy as np
from scipy.cluster.vq import vq
from recognizer import Recognizer
import cPickle as pickle

def cmd_parser():
    """
    Parse user's command line and return opts object and args
    """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-n", "--num",        help="Number of gestures", default=3, type="int")
    parser.add_option("-f", "--frames",     help="Number of frames to train on per gesture", default=100, type="int")
    parser.add_option("-w", "--words",      help="Number of visual words", default=100, type="int")
    parser.add_option("-d", "--desc",       help="Minimum number of descriptors per frame", default=100, type="int")
    parser.add_option("-t", "--type",       help="Descriptor type", action="store", type="string", default="surf")
    parser.add_option("-s", "--seed",       help="Seed for reproducibility", action="store", type="int")
    parser.add_option(      "--doc",        help="Print the docstring", action="store_true", default=False)
    parser.add_option(      "--notrain",    help="Whether to train the system", action="store_true", default=False)
    parser.add_option(      "--nocollect",  help="Whether to collect train descriptors", action="store_true", default=False)
    parser.add_option(      "--notest",     help="Whether to run the system in test mode", action="store_true", default=False)  
    return parser.parse_args()

#########################
### Main script entry ###
#########################
if __name__ == "__main__":
    opts,args = cmd_parser()
    if opts.seed is None:
        import random
        rseed = random.randint(0x00000000, 0xffffffff)
    else:
        rseed = opts.seed
    trainDirName = "Train_{0}".format(rseed)
    trainDirPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainData", trainDirName)
    if not os.path.exists(trainDirPath):
        os.makedirs(trainDirPath)
        if opts.notrain or opts.nocollect:
            print "Training data not already present for seed={0}".format(rseed)
            exit(0)
    classifierFile = os.path.join(trainDirPath, 'classifier.pkl')

    vc = cv2.VideoCapture(0)
    try:
        recognizer = Recognizer(vc=vc, descriptor='surf', opts=opts)

        if not opts.notrain:
            score = recognizer.train_from_video()   
            print "Training score = {0}".format(score)
            with open(classifierFile, 'wb') as output:
                clf = recognizer.trainer.classifier
                pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
        elif not opts.notest:
            with open(classifierFile, 'rb') as input:
                clf = pickle.load(input)

        if not opts.notest:
            recognizer.test_on_video(clf)
    except:
        vc.release()
        import traceback
        traceback.print_exc(file=sys.stdout)
    print "Seed = {0}".format(rseed)